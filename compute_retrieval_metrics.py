#!/usr/bin/env python3
# compute_retrieval_metrics.py

import argparse
import json
import math
import sqlite3
import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import re

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ============== Embedding ==============

class Embedder:
    def __init__(self, model_name: str, device: str = None, query_prefix: str = "", max_len: int = 512, last4: bool = False):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.max_len = max_len
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name,trust_remote_code=True, torch_dtype=torch.float16)
        if self.model_name == "models/jina-embeddings-v3":
            self.model[0].default_task = 'retrieval.query'
        self.model.to(self.device).eval()
        self.last4 = last4

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if self.model_name == "models/jina-embeddings-v4":
            embeddings = self.model.encode_text(
                texts, 
                batch_size=batch_size,
                task="retrieval",
                prompt_name="query",
            )
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu()
            if isinstance(embeddings, list):
                embeddings = np.array([e.cpu().numpy() for e in embeddings])
            return np.array(embeddings).astype(np.float32)
        elif self.model_name == "models/jina-embeddings-v3":
            embeddings = self.model.encode_text(
                texts, 
                batch_size=batch_size,
                task="retrieval",
                prompt_name="query",
            )   
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu()
            if isinstance(embeddings, list):
                embeddings = np.array([e.cpu().numpy() for e in embeddings])
            return np.array(embeddings).astype(np.float32)      
        else:
            out = []
            for i in range(0, len(texts), batch_size):
                batch = [self.query_prefix + t for t in texts[i:i+batch_size]]
                enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
                if self.last4:
                    hs = self.model(**enc, output_hidden_states=True).hidden_states
                    X = torch.stack(hs[-4:]).mean(0)
                else:
                    X = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).type_as(X)
                v = (X * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                v = torch.nn.functional.normalize(v, p=2, dim=1)
                out.append(v.cpu().numpy().astype(np.float32))
            return np.vstack(out)

# ============== Data loading ==============

def connect(db_path: str) -> sqlite3.Connection:
    # Check if we have a separate QA database
    qa_db_path = db_path.replace('.sqlite', '_qa.sqlite')
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    
    # Attach QA database if it exists
    import os
    if os.path.exists(qa_db_path):
        conn.execute(f"ATTACH DATABASE '{qa_db_path}' AS qa_db")
    
    return conn
def get_embedding_column_name(model_name):
    """Convert model name to valid SQLite column name"""
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', model_name.replace('/', '_').replace('-', '_'))
    return f"embedding_{clean_name}"

def load_index(conn: sqlite3.Connection, model_name: str = None):
    """
    Returns:
      E           [N, D] float32, L2-normalized embeddings
      sec_ids     [N] int section_id
      rep_ids     [N] int indices into rep_list
      rep_list    List[str] report_ids
      sec_names   List[str] section names (uppercased)
      pat_ids     [N] patient_id string per row
    """
    # Determine which embedding column to use
    if model_name:
        embedding_col = get_embedding_column_name(model_name)
    else:
        # Try to find any embedding column
        cursor = conn.execute("PRAGMA table_info(report_sections)")
        columns = [row[1] for row in cursor.fetchall()]
        embedding_cols = [col for col in columns if col.startswith('embedding_')]
        if not embedding_cols:
            raise ValueError("No embedding columns found in report_sections table")
        embedding_col = embedding_cols[0]  # Use the first one found
        print(f"Using embedding column: {embedding_col}")

    rows = conn.execute(f"""
      SELECT rs.section_id,
             rs.report_id,
             UPPER(rs.name) AS section_name,
             rs.{embedding_col} as embedding,
             r.patient_id
      FROM report_sections rs
      JOIN reports r ON r.report_id = rs.report_id
      WHERE rs.{embedding_col} IS NOT NULL
    """).fetchall()

    rep_map, rep_list = {}, []
    vecs, sec_ids, rep_ids, sec_names, pat_ids = [], [], [], [], []
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        if not np.isfinite(emb).all():
            continue
        vecs.append(emb)
        sec_ids.append(r["section_id"])
        if r["report_id"] not in rep_map:
            rep_map[r["report_id"]] = len(rep_list)
            rep_list.append(r["report_id"])
        rep_ids.append(rep_map[r["report_id"]])
        sec_names.append(r["section_name"] or "UNKNOWN")
        pat_ids.append(r["patient_id"])

    if not vecs:
        raise ValueError(f"No valid embeddings found in column {embedding_col}")
    
    E = np.vstack(vecs).astype(np.float32)
    return E, np.array(sec_ids), np.array(rep_ids), rep_list, sec_names, np.array(pat_ids, dtype=object)

def load_qa(conn: sqlite3.Connection) -> List[dict]:
    # Try to load from attached QA database first, then main database
    try:
        qas = conn.execute("""
          SELECT qi.qa_id,
                 qi.question,
                 qi.section_name  AS gold_section_name,
                 qi.answer_type,
                 COALESCE(qi.phenomena, '[]') AS phenomena_json,
                 qi.section_id    AS gold_section_id,
                 qi.report_id     AS gold_report_id,
                 r.patient_id     AS patient_id
          FROM qa_db.qa_items qi
          JOIN main.reports r ON r.report_id = qi.report_id
        """).fetchall()
    except:
        # Fallback to main database
        qas = conn.execute("""
          SELECT qi.qa_id,
                 qi.question,
                 qi.section_name  AS gold_section_name,
                 qi.answer_type,
                 COALESCE(qi.phenomena, '[]') AS phenomena_json,
                 qi.section_id    AS gold_section_id,
                 qi.report_id     AS gold_report_id,
                 r.patient_id     AS patient_id
          FROM qa_items qi
          JOIN reports r ON r.report_id = qi.report_id
        """).fetchall()

    print(f"Total QA items: {len(qas)}")
    out = []
    for r in qas:
        out.append({
            "qa_id": r["qa_id"],
            "question": r["question"],
            "gold_chunk_ids": [r["gold_section_id"]] if r["gold_section_id"] is not None else [],
            "gold_section_name": (r["gold_section_name"] or "UNKNOWN").upper(),
            "answer_type": r["answer_type"],
            "phenomena": json.loads(r["phenomena_json"]),
            "patient_id": r["patient_id"],
            "gold_report_id": r["gold_report_id"],
        })
    return out

# ============== Retrieval and metrics ==============

def scores_dense(E: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Cosine similarity if both are L2-normalized, using dot product."""
    return E @ q  # [N]

def recall_at_k(ranks: List[int], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r < k) / max(1, len(ranks))

def mrr(ranks: List[int]) -> float:
    return sum(1.0/(r+1) for r in ranks if r is not None) / max(1, len(ranks))

def ndcg_at_k(rel_lists: List[List[int]], k: int) -> float:
    # rel_lists: for each query a list of binary relevances of the top-k results
    total = 0.0
    for rel in rel_lists:
        gains = [rel[i] / math.log2(i+2) for i in range(min(k, len(rel)))]
        dcg = sum(gains)
        ideal = sorted(rel, reverse=True)
        idcg = sum(ideal[i] / math.log2(i+2) for i in range(min(k, len(ideal))))
        total += (dcg / idcg) if idcg > 0 else 0.0
    return total / max(1, len(rel_lists))

def evaluate_dense(
    conn: sqlite3.Connection,
    embedder: Embedder,
    topk: List[int] = [1, 5, 10, 20],
    restrict_same_report: bool = False,
    restrict_same_patient: bool = True,
    batch_size: int = 64,
    include_random_baseline: bool = True
):
    """
    Evaluate dense retrieval performance using the given embedder model.
    """
    embedding_col = get_embedding_column_name(embedder.model_name)
    
    def analyze_patient_statistics(conn: sqlite3.Connection, embedding_col: str):
        """
        Analyze snippets per patient statistics
        """
        rows = conn.execute(f"""
            SELECT r.patient_id, COUNT(*) as snippet_count
            FROM report_sections rs
            JOIN reports r ON r.report_id = rs.report_id
            WHERE rs.{embedding_col} IS NOT NULL
            GROUP BY r.patient_id
        """).fetchall()
        
        snippet_counts = [row["snippet_count"] for row in rows]
        
        print(f"\n=== Patient Statistics ===")
        print(f"Number of patients: {len(snippet_counts)}")
        print(f"Average snippets per patient: {np.mean(snippet_counts):.1f}")
        print(f"Maximum snippets per patient: {np.max(snippet_counts)}")
        print(f"Median snippets per patient: {np.median(snippet_counts):.1f}")
        print(f"Min snippets per patient: {np.min(snippet_counts)}")
        print(f"Total snippets: {np.sum(snippet_counts)}")
        
        return snippet_counts

    def analyze_report_statistics(conn: sqlite3.Connection, embedding_col: str):
        """
        Analyze snippets per report statistics
        """
        rows = conn.execute(f"""
            SELECT rs.report_id, COUNT(*) as snippet_count
            FROM report_sections rs
            WHERE rs.{embedding_col} IS NOT NULL
            GROUP BY rs.report_id
        """).fetchall()
        
        snippet_counts = [row["snippet_count"] for row in rows]
        
        print(f"\n=== Report Statistics ===")
        print(f"Number of reports: {len(snippet_counts)}")
        print(f"Average snippets per report: {np.mean(snippet_counts):.1f}")
        print(f"Maximum snippets per report: {np.max(snippet_counts)}")
        print(f"Median snippets per report: {np.median(snippet_counts):.1f}")
        print(f"Min snippets per report: {np.min(snippet_counts)}")
        print(f"Total snippets: {np.sum(snippet_counts)}")
        
        return snippet_counts

    def analyze_qa_distribution(conn: sqlite3.Connection):
        """
        Analyze QA distribution across patients and reports
        """
        # Try QA database first, then main database
        try:
            # QA per patient
            patient_qa = conn.execute("""
                SELECT r.patient_id, COUNT(*) as qa_count
                FROM qa_db.qa_items qi
                JOIN main.reports r ON r.report_id = qi.report_id
                GROUP BY r.patient_id
            """).fetchall()
            
            # QA per report
            report_qa = conn.execute("""
                SELECT qi.report_id, COUNT(*) as qa_count
                FROM qa_db.qa_items qi
                GROUP BY qi.report_id
            """).fetchall()
        except:
            # Fallback to main database
            patient_qa = conn.execute("""
                SELECT r.patient_id, COUNT(*) as qa_count
                FROM qa_items qi
                JOIN reports r ON r.report_id = qi.report_id
                GROUP BY r.patient_id
            """).fetchall()
            
            report_qa = conn.execute("""
                SELECT qi.report_id, COUNT(*) as qa_count
                FROM qa_items qi
                GROUP BY qi.report_id
            """).fetchall()
        
        patient_counts = [row["qa_count"] for row in patient_qa]
        report_counts = [row["qa_count"] for row in report_qa]
        
        print(f"\n=== QA Distribution Statistics ===")
        print(f"Patients with QA: {len(patient_counts)}")
        if patient_counts:
            print(f"Average QA per patient: {np.mean(patient_counts):.1f}")
            print(f"Max QA per patient: {np.max(patient_counts)}")
            print(f"Median QA per patient: {np.median(patient_counts):.1f}")
        
        print(f"Reports with QA: {len(report_counts)}")
        if report_counts:
            print(f"Average QA per report: {np.mean(report_counts):.1f}")
            print(f"Max QA per report: {np.max(report_counts)}")
            print(f"Median QA per report: {np.median(report_counts):.1f}")
        
        return patient_counts, report_counts

    # Add statistics analysis with the correct embedding column
    snippet_counts = analyze_patient_statistics(conn, embedding_col)
    report_snippet_counts = analyze_report_statistics(conn, embedding_col)
    patient_qa_counts, report_qa_counts = analyze_qa_distribution(conn)
    
    E, sec_ids, rep_ids, rep_list, sec_names, pat_ids = load_index(conn, embedder.model_name)
    qa = load_qa(conn)
    if not qa:
        print("No QA with gold chunks found.")
        return

    idx_by_sec = {int(sid): i for i, sid in enumerate(sec_ids)}
    secname_by_row = np.array(sec_names, dtype=object)

    queries = [q["question"] for q in qa]
    Q = embedder.encode(queries, batch_size=batch_size)

    gold_rows, cand_masks = [], []
    for q in qa:
        rows = [idx_by_sec[sid] for sid in q["gold_chunk_ids"] if sid in idx_by_sec]
        gold_rows.append(rows)
        mask = np.ones(len(sec_ids), dtype=bool)

        if restrict_same_patient and q.get("patient_id"):
            mask &= (pat_ids == q["patient_id"])
        if restrict_same_report and rows:
            rep_index = int(rep_ids[rows[0]])
            mask &= (rep_ids == rep_index)

        cand_masks.append(mask)
        
    all_ranks = []
    rel_lists_at_maxk = []
    top1_section_match = 0

    Kmax = max(topk)

    for i, qvec in enumerate(Q):
        cand_mask = cand_masks[i]
        scores = scores_dense(E[cand_mask], qvec)
        order = np.argsort(-scores)
        top_idx_local = order[:Kmax]
        cand_indices = np.nonzero(cand_mask)[0]
        top_rows = cand_indices[top_idx_local]

        gold_set = set(gold_rows[i])
        rr = None
        rel_list = []
        for rank, r in enumerate(top_rows):
            rel = 1 if r in gold_set else 0
            rel_list.append(rel)
            if rr is None and rel:
                rr = rank
        all_ranks.append(rr)
        rel_lists_at_maxk.append(rel_list[:Kmax])

        top1_row = top_rows[0] if len(top_rows) else None
        if top1_row is not None:
            top1_sec = secname_by_row[top1_row]
            if top1_sec == qa[i]["gold_section_name"]:
                top1_section_match += 1

    # Dense retrieval results
    print(f"\n=== Dense Retrieval Results ===")
    print(f"Queries evaluated: {len(qa)}")
    for k in topk:
        r = recall_at_k(all_ranks, k)
        nd = ndcg_at_k(rel_lists_at_maxk, k)
        print(f"Dense Recall@{k}: {r:.3f}   nDCG@{k}: {nd:.3f}")
    print(f"Dense MRR: {mrr(all_ranks):.3f}")
    print(f"Section routing accuracy@1: {top1_section_match/len(qa):.3f}")

    # Random baseline comparison
    if include_random_baseline:
        def evaluate_random_baseline(
            conn: sqlite3.Connection,
            topk: List[int] = [1, 5, 10, 20],
            restrict_same_patient: bool = True,
            num_trials: int = 5
        ):
            """
            Evaluate random retrieval baseline for comparison with dense retrieval
            """
            E, sec_ids, rep_ids, rep_list, sec_names, pat_ids = load_index(conn)
            qa = load_qa(conn)
            if not qa:
                print("No QA with gold chunks found.")
                return

            idx_by_sec = {int(sid): i for i, sid in enumerate(sec_ids)}
            
            # Prepare candidate masks (same logic as dense evaluation)
            gold_rows, cand_masks = [], []
            for q in qa:
                rows = [idx_by_sec[sid] for sid in q["gold_chunk_ids"] if sid in idx_by_sec]
                gold_rows.append(rows)
                
                mask = np.ones(len(sec_ids), dtype=bool)
                if restrict_same_patient and q.get("patient_id"):
                    mask &= (pat_ids == q["patient_id"])
                if restrict_same_report and rows:
                    rep_index = int(rep_ids[rows[0]])
                    mask &= (rep_ids == rep_index)
                cand_masks.append(mask)
            
            print(f"\n=== Random Baseline Evaluation ===")
            print(f"Number of trials: {num_trials}")
            
            all_trial_results = []
            
            for trial in range(num_trials):
                random.seed(42 + trial)  # Reproducible randomness
                
                all_ranks = []
                rel_lists_at_maxk = []
                Kmax = max(topk)
                
                for i in range(len(qa)):
                    cand_mask = cand_masks[i]
                    cand_indices = np.nonzero(cand_mask)[0]
                    
                    # Random ordering of candidates
                    random_order = list(range(len(cand_indices)))
                    random.shuffle(random_order)
                    top_idx_local = random_order[:Kmax]
                    top_rows = cand_indices[top_idx_local]
                    
                    # Calculate rank of first relevant
                    gold_set = set(gold_rows[i])
                    rr = None
                    rel_list = []
                    for rank, r in enumerate(top_rows):
                        rel = 1 if r in gold_set else 0
                        rel_list.append(rel)
                        if rr is None and rel:
                            rr = rank
                    all_ranks.append(rr)
                    rel_lists_at_maxk.append(rel_list[:Kmax])
                
                # Calculate metrics for this trial
                trial_results = {}
                for k in topk:
                    trial_results[f"recall@{k}"] = recall_at_k(all_ranks, k)
                    trial_results[f"ndcg@{k}"] = ndcg_at_k(rel_lists_at_maxk, k)
                trial_results["mrr"] = mrr(all_ranks)
                
                all_trial_results.append(trial_results)
            
            # Average across trials
            print(f"Queries evaluated: {len(qa)}")
            for k in topk:
                recalls = [t[f"recall@{k}"] for t in all_trial_results]
                ndcgs = [t[f"ndcg@{k}"] for t in all_trial_results]
                print(f"Random Recall@{k}: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
                print(f"Random nDCG@{k}: {np.mean(ndcgs):.3f} ± {np.std(ndcgs):.3f}")
            
            mrrs = [t["mrr"] for t in all_trial_results]
            print(f"Random MRR: {np.mean(mrrs):.3f} ± {np.std(mrrs):.3f}")
            
            return all_trial_results

        random_results = evaluate_random_baseline(conn, topk, restrict_same_patient)
        
        print(f"\n=== Dense vs Random Comparison ===")
        for k in topk:
            dense_recall = recall_at_k(all_ranks, k)
            random_recalls = [t[f"recall@{k}"] for t in random_results]
            improvement = dense_recall / np.mean(random_recalls) if np.mean(random_recalls) > 0 else float('inf')
            print(f"Recall@{k} improvement: {improvement:.2f}x ({dense_recall:.3f} vs {np.mean(random_recalls):.3f})")

    # Per-section breakdown of Recall@10
    per_sec = defaultdict(list)
    for i, rr in enumerate(all_ranks):
        sec = qa[i]["gold_section_name"]
        per_sec[sec].append(rr)
    print("\nPer-section Recall@10:")
    for sec, ranks in sorted(per_sec.items(), key=lambda x: x[0]):
        print(f"{sec:12s}  {recall_at_k(ranks, 10):.3f}  (n={len(ranks)})")

    # Per-phenomena breakdown of Recall@10
    per_ph = defaultdict(list)
    for i, rr in enumerate(all_ranks):
        tags = qa[i]["phenomena"] or []
        if not tags:
            per_ph["NONE"].append(rr)
        else:
            for t in tags:
                per_ph[t.upper()].append(rr)
    print("\nPer-phenomena Recall@10:")
    for ph, ranks in sorted(per_ph.items(), key=lambda x: x[0]):
        print(f"{ph:14s}  {recall_at_k(ranks, 10):.3f}  (n={len(ranks)})")

# ============== CLI ==============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--model-name", help="HF model used for query embeddings (or --compare-all)")
    ap.add_argument("--compare-all", action="store_true", help="Compare all available embedding models")
    ap.add_argument("--device", default=None)
    ap.add_argument("--query-prefix", default="", help='Use "query: " for E5')
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--last4", action="store_true", help="Average last 4 hidden states before pooling")
    ap.add_argument("--restrict-same-report", action="store_true", help="Evaluate within the same report as the gold")
    ap.add_argument("--topk", default="1,5,10,20")
    ap.add_argument("--restrict-same-patient", action="store_true", help="Evaluate within the same patient as the gold")
    ap.add_argument("--no-random-baseline", action="store_true", help="Skip random baseline evaluation")
    args = ap.parse_args()

    topk = [int(x) for x in args.topk.split(",")]
    conn = connect(args.db)
    
    try:
        if args.compare_all:
            evaluate_all_models(
                conn,
                topk=topk,
                restrict_same_report=args.restrict_same_report,
                restrict_same_patient=args.restrict_same_patient,
                include_random_baseline=not args.no_random_baseline
            )
        elif args.model_name:
            emb = Embedder(args.model_name, device=args.device, query_prefix=args.query_prefix, max_len=args.max_len, last4=args.last4)
            evaluate_dense(
                conn, emb, 
                topk=topk, 
                restrict_same_report=args.restrict_same_report, 
                restrict_same_patient=args.restrict_same_patient,
                include_random_baseline=not args.no_random_baseline
            )
        else:
            print("Either --model-name or --compare-all must be specified")
    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        conn.close()

def get_all_embedding_columns(conn: sqlite3.Connection) -> List[str]:
    """Get all embedding column names in the database"""
    cursor = conn.execute("PRAGMA table_info(report_sections)")
    columns = [row[1] for row in cursor.fetchall()]
    embedding_cols = [col for col in columns if col.startswith('embedding_')]
    return embedding_cols

def get_model_name_from_column(embedding_col: str) -> str:
    """Convert embedding column name back to model name"""
    # Remove 'embedding_' prefix and convert back
    clean_name = embedding_col.replace('embedding_', '')
    
    # More comprehensive mapping
    if 'models_multilingual_e5_large' in clean_name:
        return 'models/multilingual-e5-large'
    elif 'models_ClinicalBERT' in clean_name:
        return 'models/ClinicalBERT'
    elif 'intfloat_multilingual_e5_large' in clean_name:
        return 'intfloat/multilingual-e5-large'
    elif 'sentence_transformers' in clean_name:
        # Handle sentence-transformers models
        parts = clean_name.split('_')
        return '/'.join(parts[:2]) + '/' + '-'.join(parts[2:])
    else:
        # Generic fallback - try to reconstruct from underscores
        parts = clean_name.split('_')
        if len(parts) >= 2:
            # Assume format like: org_model_name or org_model_name_variant
            return f"{parts[0]}/{'-'.join(parts[1:])}"
        else:
            return clean_name

def evaluate_all_models(
    conn: sqlite3.Connection,
    topk: List[int] = [1, 5, 10, 20],
    restrict_same_report: bool = False,
    restrict_same_patient: bool = True,
    include_random_baseline: bool = True
):
    """
    Evaluate all available embedding models and compare their performance
    """
    embedding_cols = get_all_embedding_columns(conn)
    if not embedding_cols:
        print("No embedding columns found in database")
        return
    
    print(f"Found {len(embedding_cols)} embedding models:")
    for col in embedding_cols:
        print(f"  - {col}")
    print()
    
    qa = load_qa(conn)
    if not qa:
        print("No QA with gold chunks found.")
        return
    
    all_results = {}
    
    # Evaluate each model
    for embedding_col in embedding_cols:
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {embedding_col}")
        print(f"{'='*60}")
        
        try:
            # Load index for this specific embedding column
            E, sec_ids, rep_ids, rep_list, sec_names, pat_ids = load_index_by_column(conn, embedding_col)
            
            if len(E) == 0:
                print(f"No embeddings found for {embedding_col}")
                continue
                
            print(f"Loaded {len(E)} embeddings for {embedding_col}")
            
            # Use dummy embedder since we're loading precomputed embeddings
            dummy_embedder = DummyEmbedder(embedding_col)
            
            results = evaluate_dense_precomputed(
                conn, E, sec_ids, rep_ids, rep_list, sec_names, pat_ids, qa,
                embedding_col=embedding_col,
                topk=topk,
                restrict_same_report=restrict_same_report,
                restrict_same_patient=restrict_same_patient,
                include_random_baseline=include_random_baseline
            )
            
            all_results[embedding_col] = results
            
        except Exception as e:
            print(f"Error evaluating {embedding_col}: {e}")
            continue
    
    # Print comparison table
    print_comparison_table(all_results, topk)

def load_index_by_column(conn: sqlite3.Connection, embedding_col: str):
    """Load index using a specific embedding column"""
    rows = conn.execute(f"""
      SELECT rs.section_id,
             rs.report_id,
             UPPER(rs.name) AS section_name,
             rs.{embedding_col} as embedding,
             r.patient_id
      FROM report_sections rs
      JOIN reports r ON r.report_id = rs.report_id
      WHERE rs.{embedding_col} IS NOT NULL
    """).fetchall()

    rep_map, rep_list = {}, []
    vecs, sec_ids, rep_ids, sec_names, pat_ids = [], [], [], [], []
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        if not np.isfinite(emb).all():
            continue
        vecs.append(emb)
        sec_ids.append(r["section_id"])
        if r["report_id"] not in rep_map:
            rep_map[r["report_id"]] = len(rep_list)
            rep_list.append(r["report_id"])
        rep_ids.append(rep_map[r["report_id"]])
        sec_names.append(r["section_name"] or "UNKNOWN")
        pat_ids.append(r["patient_id"])

    if not vecs:
        raise ValueError(f"No valid embeddings found in column {embedding_col}")
    
    E = np.vstack(vecs).astype(np.float32)
    return E, np.array(sec_ids), np.array(rep_ids), rep_list, sec_names, np.array(pat_ids, dtype=object)

class DummyEmbedder:
    """Dummy embedder for precomputed embeddings"""
    def __init__(self, model_name: str):
        self.model_name = model_name

def evaluate_dense_precomputed(
    conn: sqlite3.Connection,
    E: np.ndarray,
    sec_ids: np.ndarray,
    rep_ids: np.ndarray,
    rep_list: List[str],
    sec_names: List[str],
    pat_ids: np.ndarray,
    qa: List[dict],
    embedding_col: str,
    topk: List[int] = [1, 5, 10, 20],
    restrict_same_report: bool = False,
    restrict_same_patient: bool = True,
    include_random_baseline: bool = False
):
    """Evaluate using precomputed embeddings - but we still need to encode queries"""
    
    # We need to determine the model name from the embedding column to create an embedder
    model_name = get_model_name_from_column(embedding_col)
    
    try:
        # Create embedder for query encoding
        embedder = Embedder(model_name, query_prefix="query: " if "e5" in model_name.lower() else "")
        print(f"Created embedder for {model_name}")
    except Exception as e:
        print(f"Could not create embedder for {model_name}: {e}")
        # Return placeholder results
        results = {
            'queries_evaluated': len(qa),
            'queries_with_gold': 0,
            'embedding_count': len(E),
            'embedding_col': embedding_col
        }
        for k in topk:
            results[f'recall@{k}'] = 0.0
            results[f'ndcg@{k}'] = 0.0
        results['mrr'] = 0.0
        return results
    
    idx_by_sec = {int(sid): i for i, sid in enumerate(sec_ids)}
    secname_by_row = np.array(sec_names, dtype=object)

    # Encode queries using the embedder
    queries = [q["question"] for q in qa]
    Q = embedder.encode(queries, batch_size=32)
    
    gold_rows, cand_masks = [], []
    for q in qa:
        rows = [idx_by_sec[sid] for sid in q["gold_chunk_ids"] if sid in idx_by_sec]
        gold_rows.append(rows)
        mask = np.ones(len(sec_ids), dtype=bool)

        if restrict_same_patient and q.get("patient_id"):
            mask &= (pat_ids == q["patient_id"])
        if restrict_same_report and rows:
            rep_index = int(rep_ids[rows[0]])
            mask &= (rep_ids == rep_index)

        cand_masks.append(mask)
    
    # Perform retrieval evaluation (same as evaluate_dense)
    all_ranks = []
    rel_lists_at_maxk = []
    top1_section_match = 0
    Kmax = max(topk)

    for i, qvec in enumerate(Q):
        cand_mask = cand_masks[i]
        scores = scores_dense(E[cand_mask], qvec)
        order = np.argsort(-scores)
        top_idx_local = order[:Kmax]
        cand_indices = np.nonzero(cand_mask)[0]
        top_rows = cand_indices[top_idx_local]

        gold_set = set(gold_rows[i])
        rr = None
        rel_list = []
        for rank, r in enumerate(top_rows):
            rel = 1 if r in gold_set else 0
            rel_list.append(rel)
            if rr is None and rel:
                rr = rank
        all_ranks.append(rr)
        rel_lists_at_maxk.append(rel_list[:Kmax])

        top1_row = top_rows[0] if len(top_rows) else None
        if top1_row is not None:
            top1_sec = secname_by_row[top1_row]
            if top1_sec == qa[i]["gold_section_name"]:
                top1_section_match += 1

    # Calculate metrics
    results = {
        'queries_evaluated': len(qa),
        'queries_with_gold': sum(1 for gr in gold_rows if gr),
        'embedding_count': len(E),
        'embedding_col': embedding_col
    }
    
    for k in topk:
        results[f'recall@{k}'] = recall_at_k(all_ranks, k)
        results[f'ndcg@{k}'] = ndcg_at_k(rel_lists_at_maxk, k)
    results['mrr'] = mrr(all_ranks)
    results['section_accuracy'] = top1_section_match / len(qa) if qa else 0.0
    
    # Print results for this model
    print(f"Queries evaluated: {len(qa)}")
    for k in topk:
        r = results[f'recall@{k}']
        nd = results[f'ndcg@{k}']
        print(f"Recall@{k}: {r:.3f}   nDCG@{k}: {nd:.3f}")
    print(f"MRR: {results['mrr']:.3f}")
    print(f"Section routing accuracy@1: {results['section_accuracy']:.3f}")
    
    return results

def print_comparison_table(all_results: Dict[str, dict], topk: List[int]):
    """Print a comparison table of all models"""
    print(f"\n{'='*150}")
    print("MODEL COMPARISON")
    print(f"{'='*150}")
    
    if not all_results:
        print("No results to compare")
        return
    
    # Header with consistent spacing
    header = f"{'Model':<45}"
    for k in topk:
        header += f" {'R@'+str(k):<7}"  # Fixed spacing to match data
    for k in topk:
        header += f" {'nDCG@'+str(k):<7}"  # Fixed spacing to match data
    header += f" {'MRR':<8} {'SecAcc':<8} {'Queries':<8}"
    print(header)
    print("-" * len(header))  # Make separator line match header length
    
    # Sort by best Recall@10 or first available metric
    def get_sort_key(item):
        _, results = item
        if 'recall@10' in results:
            return results['recall@10']
        elif topk and f'recall@{topk[0]}' in results:
            return results[f'recall@{topk[0]}']
        return 0
    
    sorted_results = sorted(all_results.items(), key=get_sort_key, reverse=True)
    
    for model_name, results in sorted_results:
        # Shorten model name for display
        display_name = model_name.replace('embedding_', '').replace('models_', '')[:44]  # Leave space for padding
        row = f"{display_name:<45}"
        
        # Recall metrics - match header spacing exactly
        for k in topk:
            recall = results.get(f'recall@{k}', 0.0)
            row += f" {recall:<7.3f}"
        
        # nDCG metrics - match header spacing exactly
        for k in topk:
            ndcg = results.get(f'ndcg@{k}', 0.0)
            row += f" {ndcg:<7.3f}"
        
        mrr = results.get('mrr', 0.0)
        sec_acc = results.get('section_accuracy', 0.0)
        queries = results.get('queries_evaluated', 0)
        row += f" {mrr:<8.3f} {sec_acc:<8.3f} {queries:<8}"
        
        print(row)

# Add command line option for comparing all models
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--model-name", help="HF model used for query embeddings (or --compare-all)")
    ap.add_argument("--compare-all", action="store_true", help="Compare all available embedding models")
    ap.add_argument("--device", default=None)
    ap.add_argument("--query-prefix", default="", help='Use "query: " for E5')
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--last4", action="store_true", help="Average last 4 hidden states before pooling")
    ap.add_argument("--restrict-same-report", action="store_true", help="Evaluate within the same report as the gold")
    ap.add_argument("--topk", default="1,5,10,20")
    ap.add_argument("--restrict-same-patient", action="store_true", help="Evaluate within the same patient as the gold")
    ap.add_argument("--no-random-baseline", action="store_true", help="Skip random baseline evaluation")
    args = ap.parse_args()

    topk = [int(x) for x in args.topk.split(",")]
    conn = connect(args.db)
    
    try:
        if args.compare_all:
            evaluate_all_models(
                conn,
                topk=topk,
                restrict_same_report=args.restrict_same_report,
                restrict_same_patient=args.restrict_same_patient,
                include_random_baseline=not args.no_random_baseline
            )
        elif args.model_name:
            emb = Embedder(args.model_name, device=args.device, query_prefix=args.query_prefix, max_len=args.max_len, last4=args.last4)
            evaluate_dense(
                conn, emb, 
                topk=topk, 
                restrict_same_report=args.restrict_same_report, 
                restrict_same_patient=args.restrict_same_patient,
                include_random_baseline=not args.no_random_baseline
            )
        else:
            print("Either --model-name or --compare-all must be specified")
    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
