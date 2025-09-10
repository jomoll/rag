#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunk-level QA generation and validation pipeline.

Reads chunks from `report_sections` in a SQLite DB, prompts a local Llama 3.3 70B endpoint
via an OpenAI-compatible /chat/completions API, validates outputs, and writes into:
  - qa_items            one row per accepted QA item
  - qa_gold_chunks      map of QA item to one or more acceptable gold chunk ids

Run:
    python generate_chunk_qa.py \
        --db myeloma_reports.sqlite \
        --limit 500 \
        --max-per-chunk 2 \
        --evaluation_model llama-3.3-70b
"""

import argparse
import asyncio
import contextlib
import dataclasses
import json
import random
import re
import sqlite3
import string
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging

import httpx
import numpy as np

# ---------------------------
# Configuration
# ---------------------------

SYSTEM_PROMPT = (
    "Du erstellst extraktive QA-Paare aus einem einzelnen Abschnitt eines Radiologie-Berichts.\n"
    "Du darfst nur Fragen stellen, die allein aus diesem Abschnitt beantwortet werden können.\n"
    "Antworten müssen exakte Textausschnitte aus dem Abschnitt mit Zeichenoffsets sein.\n"
    "Verwende keine Informationen außerhalb dieses Abschnitts.\n"
)

USER_PROMPT_TMPL = """Abschnitt: {section_name}

Abschnittstext:
<<<
{chunk_text}
>>>

Erstelle bis zu {max_items} QA-Elemente als JSON-Array. Jedes Element muss ein Objekt mit folgenden Schlüsseln sein:
- question (Frage auf Deutsch)
- answer_type in ["span","boolean","numeric","classification","list"]
- start_char ganzzahliger Offset in den Abschnittstext
- end_char ganzzahliger Offset in den Abschnittstext (exklusiv)
- answer_text exakt gleich chunk_text[start_char:end_char]
- phenomena Liste von Tags wie ["numeric","negation","comparison","entity","section_intent"]
- difficulty in ["easy","medium","hard"]

Einschränkungen:
1) Der answer_text muss exakt chunk_text[start_char:end_char] entsprechen.
2) Wenn der Abschnitt eine Zahl enthält, füge mindestens ein numerisches Element hinzu.
3) Wenn der Abschnitt eine Verneinung wie "kein", "ohne", "fehlt" oder "Ausschluss" enthält, füge ein Element hinzu, das darauf abzielt.
4) Vermeide triviale Fragen, die nur die Überschrift wiederholen. Frage nach konkretem Inhalt.
5) Keine Verweise auf Informationen außerhalb dieses Abschnitts. Keine Pronomen, die früheren Kontext erfordern.
6) Alle Fragen müssen auf Deutsch formuliert werden.

Gib nur JSON zurück. Keine Code-Blöcke verwenden.
"""

DEFAULT_MODEL_NAME = "llama-3.3-70b-instruct"
PROMPT_VERSION = "v1.0-de"

# German phenomena detection helpers
RE_NUMBER = re.compile(r"\d")
RE_NEGATION = re.compile(r"\b(kein|keine|keiner|ohne|fehlt|fehlend|Ausschluss|negativ|nicht|unauffällig)\b", re.I)
RE_COMPARISON = re.compile(r"\b(verglichen (mit|zu)|seit|gegenüber|im Vergleich|Verlauf|Veränderung)\b", re.I)

MIN_Q_TOKENS = 8                       # reject very short questions
MAX_SPAN_CHARS = 320                   # do not accept very long spans
MIN_SPAN_CHARS = 3

# ---------------------------
# Data access
# ---------------------------
# Add retry configuration
@dataclasses.dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    exponential_base: float = 2.0
    jitter: bool = True
    
    # HTTP status codes that should trigger retries
    retryable_status_codes: set = dataclasses.field(default_factory=lambda: {
        424,  # Failed Dependency
        429,  # Rate limit
        500, 502, 503, 504,  # Server errors
        408,  # Request timeout
    })
    
    # Exceptions that should trigger retries
    retryable_exceptions: tuple = (
        httpx.TimeoutException,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.NetworkError,
        httpx.RemoteProtocolError,
        TypeError,
        ValueError,
        httpx.HTTPStatusError,
    )

# Enhanced logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def ensure_schema_fresh(conn: sqlite3.Connection) -> None:
    """Drop and recreate tables with new schema."""
    conn.executescript("""
    DROP TABLE IF EXISTS qa_gold_chunks;
    DROP TABLE IF EXISTS qa_items;
    
    CREATE TABLE qa_items (
      qa_id INTEGER PRIMARY KEY AUTOINCREMENT,
      section_id INTEGER NOT NULL,                 -- FK to report_sections.section_id
      report_id TEXT NOT NULL,
      section_name TEXT NOT NULL,
      chunk_index INTEGER NOT NULL,
      question TEXT NOT NULL,
      answer_text TEXT NOT NULL,
      start_char INTEGER NOT NULL,
      end_char INTEGER NOT NULL,
      answer_type TEXT NOT NULL,
      phenomena TEXT,                              -- JSON array of tags
      difficulty TEXT,
      generator_model TEXT NOT NULL,
      prompt_version TEXT NOT NULL,
      valid_offsets INTEGER NOT NULL DEFAULT 1,   -- 1 if offsets are valid, 0 if not
      validation_status TEXT NOT NULL DEFAULT 'valid',  -- 'valid' or specific error code
      validation_errors TEXT,                     -- JSON array of validation error details
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(section_id, question)
    );
    CREATE INDEX idx_qa_section ON qa_items(section_id);
    CREATE INDEX idx_qa_report ON qa_items(report_id);
    CREATE INDEX idx_qa_validation ON qa_items(validation_status);

    CREATE TABLE qa_gold_chunks (
      qa_id INTEGER NOT NULL,
      section_id INTEGER NOT NULL,                 -- acceptable gold chunk id
      PRIMARY KEY (qa_id, section_id),
      FOREIGN KEY (qa_id) REFERENCES qa_items(qa_id) ON DELETE CASCADE
    );
    """)
    conn.commit()
    
@dataclasses.dataclass
class Chunk:
    section_id: int
    report_id: str
    section_name: str
    chunk_index: int
    text: str
    prev_text: Optional[str]
    next_text: Optional[str]
    prev_section_id: Optional[int]
    next_section_id: Optional[int]

def get_candidate_chunks(conn: sqlite3.Connection, limit: int, min_len: int = 60, max_len: int = 3000,
                         balance_by_section: bool = True) -> List[Chunk]:
    """
    Pull candidate chunks that do not already have QA, with optional balancing by section name.
    """
    # all section names present
    rows = conn.execute("""
        SELECT UPPER(name) AS section_name, COUNT(*) AS n
        FROM report_sections
        WHERE LENGTH(text) BETWEEN ? AND ?
    """, (min_len, max_len)).fetchall()

    section_names = {r["section_name"] for r in rows if r["section_name"]}
    per_section = max(1, limit // max(1, len(section_names))) if balance_by_section else limit

    results: List[Chunk] = []
    for sname in sorted(section_names):
        cur = conn.execute("""
          SELECT rs.section_id, rs.report_id, rs.name AS section_name, rs.chunk_index, rs.text
          FROM report_sections rs
          WHERE UPPER(rs.name)=? AND LENGTH(rs.text) BETWEEN ? AND ?
            AND NOT EXISTS (
              SELECT 1 FROM qa_items qi WHERE qi.section_id = rs.section_id
            )
          ORDER BY RANDOM()
          LIMIT ?
        """, (sname, min_len, max_len, per_section)).fetchall()

        for r in cur:
            # neighbor information for overlap checks
            prev_row = conn.execute("""
               SELECT section_id, text FROM report_sections
               WHERE report_id=? AND UPPER(name)=UPPER(?) AND chunk_index=?
            """, (r["report_id"], r["section_name"], r["chunk_index"] - 1)).fetchone()
            next_row = conn.execute("""
               SELECT section_id, text FROM report_sections
               WHERE report_id=? AND UPPER(name)=UPPER(?) AND chunk_index=?
            """, (r["report_id"], r["section_name"], r["chunk_index"] + 1)).fetchone()

            results.append(Chunk(
                section_id=r["section_id"],
                report_id=r["report_id"],
                section_name=r["section_name"],
                chunk_index=r["chunk_index"],
                text=r["text"],
                prev_text=prev_row["text"] if prev_row else None,
                next_text=next_row["text"] if next_row else None,
                prev_section_id=prev_row["section_id"] if prev_row else None,
                next_section_id=next_row["section_id"] if next_row else None,
            ))
    # cap to requested limit
    random.shuffle(results)
    return results[:limit]

# ---------------------------
# LLM client
# ---------------------------
async def call_llm_json_with_retry(client: httpx.AsyncClient, prompt_text: str, model_name: str, 
                                   retry_config: RetryConfig = None) -> List[Dict[str, Any]]:
    """
    Call LLM with exponential backoff retry logic.
    """
    if retry_config is None:
        retry_config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(retry_config.max_retries + 1):
        try:
            return await call_llm_json(client, prompt_text, model_name)
            
        except httpx.HTTPStatusError as e:
            last_exception = e
            if e.response.status_code not in retry_config.retryable_status_codes:
                logger.error(f"Non-retryable HTTP error {e.response.status_code}: {e}")
                raise
            
            if attempt == retry_config.max_retries:
                logger.error(f"Max retries ({retry_config.max_retries}) exceeded for HTTP {e.response.status_code}")
                raise
                
            delay = calculate_delay(attempt, retry_config)
            logger.warning(f"HTTP {e.response.status_code} error, retrying in {delay:.2f}s (attempt {attempt + 1}/{retry_config.max_retries + 1})")
            await asyncio.sleep(delay)
            
        except retry_config.retryable_exceptions as e:
            last_exception = e
            if attempt == retry_config.max_retries:
                logger.error(f"Max retries ({retry_config.max_retries}) exceeded for {type(e).__name__}: {e}")
                raise
                
            delay = calculate_delay(attempt, retry_config)
            logger.warning(f"{type(e).__name__} error, retrying in {delay:.2f}s (attempt {attempt + 1}/{retry_config.max_retries + 1}): {e}")
            await asyncio.sleep(delay)
            
        except json.JSONDecodeError as e:
            last_exception = e
            if attempt == retry_config.max_retries:
                logger.error(f"Max retries ({retry_config.max_retries}) exceeded for JSON decode error: {e}")
                return []  # Return empty list for JSON errors instead of raising
                
            delay = calculate_delay(attempt, retry_config)
            logger.warning(f"JSON decode error, retrying in {delay:.2f}s (attempt {attempt + 1}/{retry_config.max_retries + 1}): {e}")
            await asyncio.sleep(delay)
            
        except Exception as e:
            # For unexpected exceptions, don't retry
            logger.error(f"Unexpected error (not retrying): {type(e).__name__}: {e}")
            raise
    
    # This should never be reached, but just in case
    if last_exception:
        raise last_exception
    return []

def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay with constant backoff and optional jitter."""
    delay = config.base_delay
    
    if config.jitter:
        # Add ±25% jitter to prevent thundering herd
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)
        delay = max(0, delay)  # Ensure non-negative
    
    return delay

async def call_llm_json(client: httpx.AsyncClient, prompt_text: str, model_name: str) -> List[Dict[str, Any]]:
    """Original LLM call function."""
    payload = generate_llama_payload(prompt_text, model=model_name)
    
    # Log the request for debugging
    logger.debug(f"Making LLM request with model {model_name}, prompt length: {len(prompt_text)}")
    
    # Make the request
    resp = await client.post("/chat/completions", json=payload)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    
    logger.debug(f"LLM response length: {len(content)}")
    
    content = normalize_json_text(content)
    try:
        arr = json.loads(content)
        if isinstance(arr, dict):
            arr = [arr]
        if not isinstance(arr, list):
            logger.warning("LLM returned non-list JSON, returning empty list")
            return []
        # keep only dict items
        result = [x for x in arr if isinstance(x, dict)]
        logger.debug(f"Parsed {len(result)} valid items from LLM response")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response: {e}")
        logger.error(f"Response content: {content[:500]}...")  # Log first 500 chars
        raise

async def generate_for_chunk(client: httpx.AsyncClient, model_name: str, chunk: Chunk,
                             max_items: int, retry_config: RetryConfig = None) -> List[Dict[str, Any]]:
    """Generate QA items for a chunk with retry logic."""
    prompt = USER_PROMPT_TMPL.format(
        section_name=chunk.section_name,
        chunk_text=chunk.text,
        max_items=max_items
    )
    
    logger.info(f"Generating QA for section_id={chunk.section_id}, chunk_index={chunk.chunk_index}")
    
    try:
        items = await call_llm_json_with_retry(client, prompt, model_name, retry_config)
        validated = validate_items_for_chunk(chunk, items)
        logger.info(f"Generated {len(items)} items, {len(validated)} passed validation for section_id={chunk.section_id}")
        return validated
    except Exception as e:
        logger.error(f"Failed to generate QA for section_id={chunk.section_id}: {type(e).__name__}: {e}")
        return []

# Enhanced client creation with better timeout and retry settings
def make_async_client(evaluation_model: str) -> Tuple[httpx.AsyncClient, str]:
    """
    Returns an async client with improved timeout and connection settings.
    """
    if evaluation_model == "llama-3.3-70b":
        BASE_URL = "http://localhost:9999/v1"
        TGI_TOKEN = "dacbebe8c973154018a3d0f5"
        HEADERS = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TGI_TOKEN}",
        }
        # More generous timeouts for LLM calls
        TIMEOUT = httpx.Timeout(connect=15.0, read=180.0, write=15.0, pool=15.0)
        HTTP2 = False
        # Allow more connections for better throughput
        LIMITS = httpx.Limits(max_connections=5, max_keepalive_connections=2)
        client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers=HEADERS,
            timeout=TIMEOUT,
            http2=HTTP2,
            limits=LIMITS
        )
        model_name = DEFAULT_MODEL_NAME
        return client, model_name

    raise ValueError(f"Unknown evaluation_model={evaluation_model}")


def generate_llama_payload(prompt_text: str, model: str = DEFAULT_MODEL_NAME, max_tokens: int = 800,
                           temperature: float = 0.3, top_p: float = 0.95) -> Dict[str, Any]:
    """
    OpenAI-compatible chat payload.
    """
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": 1,
        "stop": None,
        "stream": False,
    }

# ---------------------------
# Validation
# ---------------------------

def count_tokens_like(s: str) -> int:
    # very rough heuristic token count
    return len(s.split())

def normalize_json_text(s: str) -> str:
    # remove code fences if model returns them
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        # after stripping fence markers, attempt to find first [ and last ]
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            s = s[start:end+1]
    return s

def tag_phenomena(chunk_text: str, answer_text: str, existing: Optional[List[str]]) -> List[str]:
    tags = set(existing or [])
    if RE_NUMBER.search(answer_text):
        tags.add("numeric")
    if RE_NEGATION.search(answer_text) or RE_NEGATION.search(chunk_text):
        tags.add("negation")
    if RE_COMPARISON.search(chunk_text):
        tags.add("comparison")
    # cheap proxy
    if answer_text.isupper() or len(answer_text.split()) <= 2:
        pass
    return sorted(tags)

# Update the German trivial question patterns
def looks_trivial_question(q: str, section_name: str) -> bool:
    ql = q.lower().strip()
    if len(ql) == 0:
        return True
    if count_tokens_like(ql) < MIN_Q_TOKENS:
        return True
    # German trivial patterns
    trivial = [
        "was ist der eindruck",
        "was ist die beurteilung", 
        "was sind die befunde",
        "was ist die indikation",
        "was ist die technik",
        "was ist die empfehlung",
        "wie lautet der befund",
        "wie lautet die diagnose",
        f"was ist {section_name.lower()}",
        f"was steht in {section_name.lower()}",
    ]
    return any(p in ql for p in trivial)

def validate_items_for_chunk(chunk: Chunk, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate items and add validation status/errors instead of filtering out invalid ones.
    """
    results = []
    text = chunk.text

    for it in items:
        validation_errors = []
        validation_status = "valid"
        valid_offsets = 1
        
        try:
            q = it.get("question", "").strip()
            a_type = it.get("answer_type", "").strip()
            start = it.get("start_char")
            end = it.get("end_char")
            ans = it.get("answer_text", "")
            phen = it.get("phenomena", []) or []
            diff = it.get("difficulty", "easy")

            # Basic type and format checks
            if not q:
                validation_errors.append("empty_question")
            
            if a_type not in {"span","boolean","numeric","classification","list"}:
                validation_errors.append(f"invalid_answer_type:{a_type}")
            
            if looks_trivial_question(q, chunk.section_name):
                validation_errors.append("trivial_question")
            
            # Offset validation
            try:
                start = int(start)
                end = int(end)
            except (ValueError, TypeError):
                validation_errors.append("invalid_offset_format")
                valid_offsets = 0
                start = 0
                end = 0
            
            if valid_offsets and (start < 0 or end <= start or end > len(text)):
                validation_errors.append(f"offset_out_of_bounds:start={start},end={end},text_len={len(text)}")
                valid_offsets = 0
            
            # Exact substring match
            if valid_offsets and text[start:end] != ans:
                actual_substr = text[start:end] if start >= 0 and end <= len(text) else ""
                validation_errors.append(f"substring_mismatch:expected='{ans}',actual='{actual_substr}'")
                valid_offsets = 0
            
            # Span length bounds
            if len(ans) < MIN_SPAN_CHARS:
                validation_errors.append(f"answer_too_short:len={len(ans)},min={MIN_SPAN_CHARS}")
            elif len(ans) > MAX_SPAN_CHARS:
                validation_errors.append(f"answer_too_long:len={len(ans)},max={MAX_SPAN_CHARS}")
            
            # Type-specific checks
            if a_type == "numeric" and not RE_NUMBER.search(ans):
                validation_errors.append("numeric_type_no_numbers")
            
            # Negation check
            if "negation" in [t.lower() for t in phen]:
                if not (RE_NEGATION.search(ans) or RE_NEGATION.search(text)):
                    validation_errors.append("negation_tag_without_negation")

            # Question length check
            if count_tokens_like(q) < MIN_Q_TOKENS:
                validation_errors.append(f"question_too_short:tokens={count_tokens_like(q)},min={MIN_Q_TOKENS}")

            # Set validation status
            if validation_errors:
                validation_status = "invalid"
            
            # Neighbor gold detection (only for valid items)
            neighbor_gold_ids: List[int] = []
            if validation_status == "valid":
                if chunk.prev_text and ans in chunk.prev_text and chunk.prev_section_id:
                    neighbor_gold_ids.append(chunk.prev_section_id)
                if chunk.next_text and ans in chunk.next_text and chunk.next_section_id:
                    neighbor_gold_ids.append(chunk.next_section_id)

            # Enrich phenomena (always do this)
            phen = tag_phenomena(text, ans, phen)

            results.append({
                "question": q,
                "answer_type": a_type,
                "start_char": start,
                "end_char": end,
                "answer_text": ans,
                "phenomena": phen,
                "difficulty": diff,
                "neighbor_golds": neighbor_gold_ids,
                "validation_status": validation_status,
                "validation_errors": validation_errors,
                "valid_offsets": valid_offsets
            })
            
        except Exception as e:
            # Catch-all for unexpected errors
            validation_errors.append(f"exception:{type(e).__name__}:{str(e)}")
            results.append({
                "question": q if 'q' in locals() else it.get("question", ""),
                "answer_type": a_type if 'a_type' in locals() else it.get("answer_type", ""),
                "start_char": start if 'start' in locals() else it.get("start_char", 0),
                "end_char": end if 'end' in locals() else it.get("end_char", 0),
                "answer_text": ans if 'ans' in locals() else it.get("answer_text", ""),
                "phenomena": phen if 'phen' in locals() else [],
                "difficulty": diff if 'diff' in locals() else "easy",
                "neighbor_golds": [],
                "validation_status": "error",
                "validation_errors": validation_errors,
                "valid_offsets": 0
            })

    return results

# ---------------------------
# Persistence
# ---------------------------
def insert_qa(conn: sqlite3.Connection, chunk: Chunk, model_name: str,
              qa: Dict[str, Any], prompt_version: str) -> Optional[int]:
    cur = conn.cursor()
    cur.execute("""
      INSERT OR IGNORE INTO qa_items
      (section_id, report_id, section_name, chunk_index,
       question, answer_text, start_char, end_char, answer_type,
       phenomena, difficulty, generator_model, prompt_version, 
       valid_offsets, validation_status, validation_errors)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        chunk.section_id, chunk.report_id, chunk.section_name, chunk.chunk_index,
        qa["question"], qa["answer_text"], qa["start_char"], qa["end_char"], qa["answer_type"],
        json.dumps(qa.get("phenomena", [])), qa.get("difficulty","easy"),
        model_name, prompt_version,
        qa.get("valid_offsets", 1), qa.get("validation_status", "valid"),
        json.dumps(qa.get("validation_errors", []))
    ))
    if cur.rowcount == 0:
        return None
    qa_id = cur.lastrowid
    
    # Only add gold chunks for valid items
    if qa.get("validation_status") == "valid":
        # self gold
        cur.execute("INSERT OR IGNORE INTO qa_gold_chunks(qa_id, section_id) VALUES (?, ?)", (qa_id, chunk.section_id))
        # neighbor golds
        for sid in qa.get("neighbor_golds", []):
            cur.execute("INSERT OR IGNORE INTO qa_gold_chunks(qa_id, section_id) VALUES (?, ?)", (qa_id, sid))
    
    conn.commit()
    return qa_id


# ---------------------------
# Main
# ---------------------------
async def main_async(args):
    conn = connect(args.db)
    ensure_schema_fresh(conn)

    chunks = get_candidate_chunks(conn, limit=args.limit, min_len=args.min_len, max_len=args.max_len,
                                  balance_by_section=not args.no_balance)
    if not chunks:
        print("No candidate chunks found that need QA.")
        return

    client, model_name = make_async_client(args.evaluation_model)
    
    # Configure retries
    retry_config = RetryConfig(
        max_retries=args.max_retries,
        base_delay=args.base_delay,
    )

    created = 0
    valid_created = 0
    invalid_created = 0
    failed = 0
    start_time = time.time()

    try:
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}: section_id={chunk.section_id}")
            
            try:
                qa_items = await generate_for_chunk(client, model_name, chunk, args.max_per_chunk, retry_config)
            except Exception as e:
                failed += 1
                logger.error(f"Final failure for section_id={chunk.section_id}: {type(e).__name__}: {e}")
                continue

            kept = 0
            valid_kept = 0
            invalid_kept = 0
            
            for qa in qa_items[:args.max_keep_per_chunk]:
                qa_id = insert_qa(conn, chunk, model_name, qa, PROMPT_VERSION)
                if qa_id is not None:
                    kept += 1
                    created += 1
                    if qa.get("validation_status") == "valid":
                        valid_kept += 1
                        valid_created += 1
                    else:
                        invalid_kept += 1
                        invalid_created += 1

            logger.info(f"Completed section_id={chunk.section_id}, kept={kept} QA items (valid={valid_kept}, invalid={invalid_kept})")
            
            # Progress update every 10 chunks
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(chunks) - i - 1) / rate if rate > 0 else 0
                logger.info(f"Progress: {i+1}/{len(chunks)} chunks, {created} total QA items ({valid_created} valid, {invalid_created} invalid), {failed} failures, ETA: {eta:.1f}s")

    finally:
        with contextlib.suppress(Exception):
            await client.aclose()
        conn.close()

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.1f}s. Created {created} QA items ({valid_created} valid, {invalid_created} invalid). Failures: {failed}.")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, help="Path to SQLite database")
    p.add_argument("--evaluation_model", default="llama-3.3-70b", help="Model switch")
    p.add_argument("--limit", type=int, default=5000, help="Max chunks to process")
    p.add_argument("--max-per-chunk", type=int, default=3, help="Ask model for up to this many per chunk")
    p.add_argument("--max-keep-per-chunk", type=int, default=3, help="Keep at most this many validated items per chunk")
    p.add_argument("--min-len", type=int, default=40, help="Min chunk length in characters")
    p.add_argument("--max-len", type=int, default=3000, help="Max chunk length in characters")
    p.add_argument("--no-balance", action="store_true", help="Do not balance by section")

    # Retry configuration
    p.add_argument("--max-retries", type=int, default=10000, help="Maximum number of retries per request")
    p.add_argument("--base-delay", type=float, default=1.0, help="Base delay between retries in seconds")
    
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
