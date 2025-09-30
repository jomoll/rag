import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Search, ShieldAlert, Clock, FileText, ChevronDown, Loader2, Copy, Check, Filter, Settings, Database, ListFilter, Plus, Download, Trash2, Info, Link as LinkIcon, ExternalLink, User, Lock, TimerReset } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetFooter, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { ENDPOINT, API_KEY, API_VERSION, DEPLOYMENT_NAME } from "@/api_keys";

/**
 * Clinical RAG UI Framework
 * Opinionated skeleton for a provenance-centric, patient-scoped RAG tool.
 *
 * Design principles
 * 1) Patient scoping is required before any retrieval.
 * 2) Minimal knobs in the main flow. Expose k and time window. Put the rest behind Advanced.
 * 3) Provenance-first. Sources are always visible. Evidence pins drive regeneration.
 * 4) Clear insufficiency. The model can return an explicit insufficient context state.
 * 5) Full audit. Every run records inputs, settings and context identifiers.
 */

// Types
export type Patient = { id: string; name: string; dob?: string };
export type DocHit = {
  id: string;
  patientId: string;
  encounterId?: string;
  docType: string;
  section?: string;
  date: string; // ISO
  score?: number;
  snippet: string;
  url?: string; // deep link to EHR or viewer
  pinned?: boolean;
};

export type RetrievalSettings = {
  k: number; // 3..30
  startDate?: Date | null;
  endDate?: Date | null;
  docTypes: string[]; // filter
  globalMode: boolean; // when true, patient filter is ignored unless cohort preset enforces it
  expandSynonyms: boolean;
  reranker: "none" | "cross-encoder" | "lexical-heuristic";
  indexVersion?: string;
};

export type RunMetadata = {
  modelId: string;
  promptTemplateHash: string;
  indexSnapshotId: string;
  latencyMs?: { retrieval?: number; generation?: number; total?: number };
};

export type RagAnswer = {
  status: "ok" | "insufficient" | "error";
  text: string;
  citedDocIds: string[];
  warnings?: string[];
};

// Configuration for Azure OpenAI GPT-5 endpoint
const LLM_CONFIG = {
  azureEndpoint: ENDPOINT, // Replace with your AZURE_ENDPOINT_5
  apiKey: API_KEY, // Replace with your OPENAI_KEY_5
  apiVersion: API_VERSION, // Replace with your OPENAI_VERSION_5
  deploymentName: DEPLOYMENT_NAME, // Replace with your GPT-5 deployment name
  timeout: 120000,
};

// Update the API configuration
const API_BASE_URL = 'http://localhost:5200/api';

// Replace the fake API layer with real database calls
const api = {
  async listPatients(q: string): Promise<Patient[]> {
    console.log('🔍 Searching for patients with query:', q);
    
    try {
      const response = await fetch(`${API_BASE_URL}/patients/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: q })
      });
      
      console.log('📡 API response status:', response.status);
      
      if (!response.ok) {
        console.warn('⚠️ API server error');
        return []; // Return empty array instead of fallback
      }
      
      const results = await response.json();
      console.log('✅ API results:', results);
      
      const patients = results.map((row: any) => ({
        id: row.id,
        name: row.name,
        dob: row.dob
      }));
      
      console.log('👥 Mapped patients:', patients);
      return patients;
    } catch (error) {
      console.error('❌ Error searching patients:', error);
      return []; // Return empty array instead of fallback
    }
  },
  
  async runRag(params: {
    question: string;
    patientId?: string;
    settings: RetrievalSettings;
    pinnedDocIds: string[];
  }): Promise<{ hits: DocHit[]; answer: RagAnswer; meta: RunMetadata }> {
    const t0 = performance.now();
    
    console.log('🔍 RAG Search starting with params:', {
      question: params.question,
      patientId: params.patientId,
      k: params.settings.k,
      docTypes: params.settings.docTypes
    });
    
    // Search database for relevant documents
    let hits: DocHit[] = [];
    try {
      const searchPayload = {
        query: params.question,
        patient_id: params.patientId,
        k: params.settings.k,
        doc_types: params.settings.docTypes.length > 0 ? params.settings.docTypes : null
      };
      
      console.log('📡 Sending document search request:', searchPayload);
      
      const response = await fetch(`${API_BASE_URL}/documents/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(searchPayload)
      });
      
      console.log('📡 Document search response status:', response.status);
      
      if (response.ok) {
        const searchResults = await response.json();
        console.log('✅ Document search results:', searchResults);
        console.log('📊 Found', searchResults.length, 'documents');
        
        // Convert to DocHit format
        hits = searchResults.map((row: any) => ({
          id: row.id,
          patientId: row.patient_id,
          encounterId: row.encounter_id,
          docType: row.doc_type,
          section: row.section,
          date: row.date,
          score: row.score,
          snippet: row.snippet,
          url: `#/document/${row.id}`,
          pinned: false
        }));
        
        console.log('🎯 Mapped document hits:', hits);
      } else {
        console.error('❌ Document search failed with status:', response.status);
        const errorText = await response.text();
        console.error('❌ Error response:', errorText);
      }
    } catch (error) {
      console.error('❌ Error searching documents:', error);
    }
    
    const retrievalMs = performance.now() - t0;
    console.log('⏱️ Document retrieval took:', retrievalMs, 'ms');
    
    // Generate answer using Azure OpenAI GPT-5 endpoint with retrieved context
    const generationStart = performance.now();
    let answer: RagAnswer;
    
    try {
      const url = `${LLM_CONFIG.azureEndpoint}/openai/deployments/${LLM_CONFIG.deploymentName}/chat/completions?api-version=${LLM_CONFIG.apiVersion}`;
      
      // Build context from retrieved documents
      const context = hits.map(hit => 
        `[${hit.docType}${hit.section ? ` - ${hit.section}` : ''} | ${hit.date.substring(0, 10)} | Score: ${hit.score?.toFixed(2)}]\n${hit.snippet}`
      ).join('\n\n---\n\n');
      
      console.log('📝 Built context for LLM:', context.length, 'characters');
      console.log('📄 Context preview:', context.substring(0, 200) + '...');
      
      const systemPrompt = context.length > 0 
        ? `You are a clinical AI assistant. Answer questions based ONLY on the provided medical records context. If the context doesn't contain sufficient information to answer the question, state that clearly. Cite specific documents when possible by mentioning the document type and date.

MEDICAL RECORDS CONTEXT:
${context}`
        : "You are a clinical AI assistant. No specific patient context is available. Provide a general medical response and clearly state the lack of patient-specific information.";

      console.log('🤖 Using system prompt with context length:', systemPrompt.length);

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'api-key': LLM_CONFIG.apiKey,
        },
        body: JSON.stringify({
          messages: [
            {
              role: "system",
              content: systemPrompt
            },
            {
              role: "user",
              content: params.question
            }
          ],
          reasoning_effort: "minimal",
        }),
        signal: AbortSignal.timeout(LLM_CONFIG.timeout)
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Azure OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const data = await response.json();
      const generatedText = data.choices?.[0]?.message?.content || "No response generated.";
      
      answer = {
        status: hits.length > 0 ? "ok" : "insufficient",
        text: generatedText,
        citedDocIds: hits.map(h => h.id), // Assume all retrieved docs are potentially cited
        warnings: hits.length === 0 ? ["No relevant documents found in patient records"] : undefined
      };
      
    } catch (error: any) {
      console.error('Azure OpenAI generation error:', error);
      answer = {
        status: "error",
        text: `Error generating response: ${error.message}`,
        citedDocIds: [],
        warnings: ["Failed to connect to Azure OpenAI endpoint"]
      };
    }
    
    const generationMs = performance.now() - generationStart;
    const meta: RunMetadata = {
      modelId: "azure-gpt-5",
      promptTemplateHash: "clinical-v1-sqlite",
      indexSnapshotId: `sqlite-fts-${params.settings.k}`,
      latencyMs: { 
        retrieval: Math.round(retrievalMs), 
        generation: Math.round(generationMs), 
        total: Math.round(retrievalMs + generationMs) 
      },
    };
    
    return { hits, answer, meta };
  },
};

function delay(ms: number) {
  return new Promise(res => setTimeout(res, ms));
}

function mockHits(patientId: string | undefined, k: number): DocHit[] {
  const base: DocHit[] = Array.from({ length: k }).map((_, i) => ({
    id: `doc-${i + 1}`,
    patientId: patientId ?? "*",
    docType: i % 3 === 0 ? "Discharge Summary" : i % 3 === 1 ? "Progress Note" : "Radiology Report",
    section: i % 3 === 2 ? "Impression" : "Assessment",
    date: new Date(Date.now() - i * 24 * 3600 * 1000 * 17).toISOString(),
    score: Math.max(0, 1 - i * 0.06),
    snippet:
      i % 3 === 2
        ? "CT chest without contrast shows no acute pulmonary process. Recommendation: follow-up in 6 months."
        : "Patient reports improved dyspnea. Continue lisinopril 10 mg daily. Monitor BP and renal function.",
    url: "#",
  }));
  return base;
}

// Utility
function formatDate(iso: string) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "2-digit" });
}

// Root component
export default function ClinicalRagApp() {
  const [patientQuery, setPatientQuery] = useState("");
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);

  const [question, setQuestion] = useState("");
  const [settings, setSettings] = useState<RetrievalSettings>({
    k: 8,
    startDate: undefined,
    endDate: undefined,
    docTypes: [],
    globalMode: false,
    expandSynonyms: false,
    reranker: "none",
    indexVersion: "idx-2025-09-15",
  });

  const [isRunning, setIsRunning] = useState(false);
  const [hits, setHits] = useState<DocHit[]>([]);
  const [answer, setAnswer] = useState<RagAnswer | null>(null);
  const [meta, setMeta] = useState<RunMetadata | null>(null);
  const [copied, setCopied] = useState(false);

  // Search patients
  useEffect(() => {
    let active = true;
    console.log('🔄 Patient search effect triggered, query:', patientQuery);
    
    // If query is empty, show all patients initially
    const searchQuery = patientQuery.trim() || '';
    
    api.listPatients(searchQuery).then(list => {
      if (active) {
        console.log('📋 Setting patients state:', list);
        setPatients(list);
      }
    });
    
    return () => {
      active = false;
    };
  }, [patientQuery]);

  // Load all patients on initial mount
  useEffect(() => {
    console.log('🚀 Component mounted, loading all patients');
    api.listPatients('').then(list => {
      console.log('🏥 Initial patients loaded:', list);
      setPatients(list);
    });
  }, []); // Empty dependency array = run once on mount

  const pinnedIds = useMemo(() => hits.filter(h => h.pinned).map(h => h.id), [hits]);

  async function run() {
    setIsRunning(true);
    try {
      const runParams = {
        question: question.trim(),
        patientId: settings.globalMode ? undefined : selectedPatient?.id,
        settings,
        pinnedDocIds: pinnedIds,
      };
      const res = await api.runRag(runParams);
      setHits(res.hits);
      setAnswer(res.answer);
      setMeta(res.meta);
    } finally {
      setIsRunning(false);
    }
  }

  function togglePin(id: string) {
    setHits(prev => prev.map(h => (h.id === id ? { ...h, pinned: !h.pinned } : h)));
  }

  function clearAll() {
    setQuestion("");
    setHits([]);
    setAnswer(null);
    setMeta(null);
  }

  async function copyWithCitations() {
    const cite = hits
      .filter(h => answer?.citedDocIds.includes(h.id))
      .map(h => `[${h.docType} • ${formatDate(h.date)}]`)
      .join("; ");
    const payload = `${answer?.text || ""}\n\nSources: ${cite}`.trim();
    await navigator.clipboard.writeText(payload);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  }

  const canRun = settings.globalMode || !!selectedPatient;

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-gray-50">
        <header className="sticky top-0 z-30 backdrop-blur bg-white/80 border-b">
          <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-3">
            <Database className="w-5 h-5" />
            <h1 className="text-lg font-semibold">Clinical RAG</h1>
            <div className="ml-auto flex items-center gap-2">
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="sm" className="gap-2"><Info className="w-4 h-4" /> Details</Button>
                </SheetTrigger>
                <SheetContent side="right" className="w-[420px]">
                  <SheetHeader>
                    <SheetTitle>Run details</SheetTitle>
                  </SheetHeader>
                  <div className="mt-4 space-y-3 text-sm">
                    <div className="grid grid-cols-2 gap-3">
                      <Detail label="Model" value={meta?.modelId || "—"} />
                      <Detail label="Prompt" value={meta?.promptTemplateHash || "—"} />
                      <Detail label="Index" value={meta?.indexSnapshotId || settings.indexVersion} />
                      <Detail label="k" value={String(settings.k)} />
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                      <Detail label="Retrieval ms" value={meta?.latencyMs?.retrieval?.toString() || "—"} />
                      <Detail label="Gen ms" value={meta?.latencyMs?.generation?.toString() || "—"} />
                      <Detail label="Total ms" value={meta?.latencyMs?.total?.toString() || "—"} />
                    </div>
                    <div className="mt-2">
                      <h4 className="text-xs font-medium uppercase text-gray-500">Audit fields</h4>
                      <ul className="mt-1 text-xs list-disc pl-4 space-y-1 text-gray-600">
                        <li>patient_id: {selectedPatient ? selectedPatient.id : settings.globalMode ? "global" : "—"}</li>
                        <li>reranker: {settings.reranker}</li>
                        <li>expandSynonyms: {String(settings.expandSynonyms)}</li>
                        <li>docTypes: {settings.docTypes.length ? settings.docTypes.join(", ") : "all"}</li>
                      </ul>
                    </div>
                  </div>
                  <SheetFooter className="mt-6">
                    <Button variant="secondary" onClick={() => window.print()} className="w-full"><Download className="w-4 h-4 mr-2" /> Export</Button>
                  </SheetFooter>
                </SheetContent>
              </Sheet>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 py-6 grid lg:grid-cols-3 gap-6">
          {/* Left column: controls */}
          <div className="lg:col-span-1 space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Patient and mode</CardTitle>
                <CardDescription>Scope queries to a specific patient or switch to global mode.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-end gap-2">
                  <div className="flex-1">
                    <Label htmlFor="patient">Patient</Label>
                    <div className="relative">
                      <Input id="patient" placeholder="Search patient by name" value={patientQuery} onChange={e => setPatientQuery(e.target.value)} />
                      <div className="absolute left-2 top-2.5 text-gray-400"><User className="w-4 h-4" /></div>
                      <div className="absolute right-2 top-2.5">
                        <Button 
                          size="sm" 
                          variant="ghost" 
                          onClick={async () => {
                            try {
                              console.log('Manual API test starting...');
                              const response = await fetch(`${API_BASE_URL}/patients/search`, {
                                method: 'POST',
                                headers: {
                                  'Content-Type': 'application/json',
                                },
                                body: JSON.stringify({ query: 'jane' })
                              });
                              console.log('Response status:', response.status);
                              const data = await response.json();
                              console.log('Response data:', data);
                              setPatients(data.map((row: any) => ({
                                id: row.id,
                                name: row.name,
                                dob: row.dob
                              })));
                            } catch (error) {
                              console.error('Manual test error:', error);
                            }
                          }}
                          className="h-6 px-2 text-xs"
                        >
                          Test
                        </Button>
                      </div>
                    </div>
                    <div className="mt-2 max-h-40 border rounded-md bg-white overflow-auto">
                      {/* DEBUG PANEL - Shows current state */}
                      <div className="p-2 bg-blue-50 border-b text-xs">
                        <strong>Debug:</strong> {patients.length} patients loaded | Query: "{patientQuery}"
                        <br />
                        API URL: {API_BASE_URL}/patients/search
                        <br />
                        Last search: {patientQuery ? 'searching...' : 'no search yet'}
                      </div>
                      
                      {patients.length === 0 && (
                        <div className="p-3 text-sm text-gray-500">
                          No patients found. Try typing a name like "jane", "john", or "samir"
                        </div>
                      )}
                      
                      {patients.map(p => (
                        <button
                          key={p.id}
                          className={`w-full text-left px-3 py-2 hover:bg-gray-50 ${selectedPatient?.id === p.id ? "bg-gray-100" : ""}`}
                          onClick={() => setSelectedPatient(p)}
                        >
                          <div className="font-medium">{p.name}</div>
                          <div className="text-xs text-gray-500">{p.id}{p.dob ? ` • DOB ${p.dob}` : ""}</div>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="flex items-center justify-between bg-gray-50 border rounded-md p-3">
                  <div>
                    <div className="text-sm font-medium">Global knowledge mode</div>
                    <div className="text-xs text-gray-500">Use only if authorized for cross-patient research.</div>
                  </div>
                  <Switch checked={settings.globalMode} onCheckedChange={v => setSettings(s => ({ ...s, globalMode: v }))} />
                </div>
                {!settings.globalMode && !selectedPatient && (
                  <div className="flex items-center gap-2 text-amber-600 text-sm"><ShieldAlert className="w-4 h-4" /> Select a patient to enable retrieval.</div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3"><CardTitle className="text-base">Retrieval settings</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <Label>Top k</Label>
                    <Badge variant="secondary">{settings.k}</Badge>
                  </div>
                  <Slider value={[settings.k]} min={3} max={30} step={1} onValueChange={([v]) => setSettings(s => ({ ...s, k: v }))} />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <DateField label="Start" value={settings.startDate} onChange={d => setSettings(s => ({ ...s, startDate: d }))} />
                  <DateField label="End" value={settings.endDate} onChange={d => setSettings(s => ({ ...s, endDate: d }))} />
                </div>

                <Accordion type="single" collapsible>
                  <AccordionItem value="adv">
                    <AccordionTrigger className="text-sm"><Settings className="w-4 h-4 mr-2" /> Advanced</AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-3">
                        <div>
                          <Label>Document types</Label>
                          <div className="flex flex-wrap gap-2 mt-2">
                            {[
                              "Discharge Summary",
                              "Progress Note",
                              "Radiology Report",
                              "Lab Result",
                              "Pathology",
                            ].map(dt => (
                              <ToggleChip
                                key={dt}
                                label={dt}
                                active={settings.docTypes.includes(dt)}
                                onToggle={() =>
                                  setSettings(s => ({
                                    ...s,
                                    docTypes: s.docTypes.includes(dt)
                                      ? s.docTypes.filter(x => x !== dt)
                                      : [...s.docTypes, dt],
                                  }))
                                }
                              />
                            ))}
                          </div>
                        </div>
                        <div className="flex items-center justify-between p-3 border rounded-md">
                          <div>
                            <div className="text-sm font-medium">Synonym expansion</div>
                            <div className="text-xs text-gray-500">Adds UMLS style synonyms to the query.</div>
                          </div>
                          <Switch checked={settings.expandSynonyms} onCheckedChange={v => setSettings(s => ({ ...s, expandSynonyms: v }))} />
                        </div>
                        <div>
                          <Label>Reranker</Label>
                          <Select value={settings.reranker} onValueChange={v => setSettings(s => ({ ...s, reranker: v as any }))}>
                            <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                            <SelectContent>
                              <SelectItem value="none">None</SelectItem>
                              <SelectItem value="cross-encoder">Cross encoder</SelectItem>
                              <SelectItem value="lexical-heuristic">Lexical heuristic</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              </CardContent>
            </Card>

          </div>

          {/* Right column: query and results */}
          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Ask a question</CardTitle>
                <CardDescription>Use only the context shown in sources. If evidence is insufficient, return that state.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex gap-2">
                  <Input
                    placeholder="e.g., What was the latest plan regarding ACE inhibitors"
                    value={question}
                    onChange={e => setQuestion(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === "Enter" && canRun && !isRunning) run();
                    }}
                  />
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span>
                        <Button onClick={run} disabled={!canRun || !question.trim() || isRunning} className="gap-2">
                          {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                          Run
                        </Button>
                      </span>
                    </TooltipTrigger>
                    <TooltipContent>Enter to run</TooltipContent>
                  </Tooltip>
                  <Button variant="outline" onClick={clearAll}><Trash2 className="w-4 h-4" /></Button>
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 xl:grid-cols-5 gap-4">
              {/* Answer panel */}
              <Card className="xl:col-span-3">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">Answer</CardTitle>
                    <div className="flex items-center gap-2">
                      {answer?.status === "insufficient" && (
                        <Badge variant="destructive" className="gap-1"><ShieldAlert className="w-3 h-3" /> Insufficient context</Badge>
                      )}
                      {answer && (
                        <Button size="sm" variant="secondary" onClick={copyWithCitations} className="gap-2">
                          {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />} Copy with citations
                        </Button>
                      )}
                    </div>
                  </div>
                  <CardDescription>
                    {meta?.latencyMs && (
                      <span className="text-xs text-gray-500">{meta.latencyMs.total} ms total</span>
                    )}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {!answer && (
                    <div className="text-sm text-gray-500">Run a query to see the answer and supporting sources.</div>
                  )}
                  {answer && (
                    <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} className="prose prose-sm max-w-none">
                      <p>{answer.text}</p>
                      {answer.warnings && answer.warnings.length > 0 && (
                        <div className="mt-4 p-3 border rounded-md bg-amber-50 text-amber-800 text-sm">{answer.warnings.join(" ")}</div>
                      )}
                    </motion.div>
                  )}
                </CardContent>
              </Card>

              {/* Sources panel */}
              <Card className="xl:col-span-2">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Sources</CardTitle>
                  <CardDescription>Pin key snippets. Regenerate to bias the answer to pins.</CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[420px] pr-3">
                    <div className="space-y-3">
                      {hits.map(h => (
                        <div key={h.id} className="border rounded-lg p-3 bg-white">
                          <div className="flex items-start justify-between gap-2">
                            <div className="min-w-0">
                              <div className="flex items-center gap-2 flex-wrap">
                                <Badge variant="outline">{h.docType}</Badge>
                                {h.section && <Badge variant="secondary">{h.section}</Badge>}
                                <span className="text-xs text-gray-500"><Clock className="w-3 h-3 inline mr-1" /> {formatDate(h.date)}</span>
                                {typeof h.score === "number" && (
                                  <span className="text-xs text-gray-400">score {h.score.toFixed(2)}</span>
                                )}
                              </div>
                              <div className="mt-1 text-sm text-gray-800 line-clamp-3">{h.snippet}</div>
                              <div className="mt-2 flex items-center gap-2">
                                <Button size="sm" variant="ghost" className="h-7 px-2" onClick={() => togglePin(h.id)}>
                                  <span className={h.pinned ? "font-semibold" : ""}>{h.pinned ? "Unpin" : "Pin"}</span>
                                </Button>
                                {h.url && (
                                  <a href={h.url} target="_blank" className="text-xs text-blue-600 inline-flex items-center gap-1">
                                    <ExternalLink className="w-3 h-3" /> Open
                                  </a>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                      {hits.length === 0 && (
                        <div className="text-sm text-gray-500">No sources yet.</div>
                      )}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          </div>
        </main>

        <footer className="border-t bg-white">
          <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-3">
            <div className="text-xs text-gray-500">This interface does not store PHI outside the clinic network. Answers are generated only from shown sources.</div>
            <div className="ml-auto flex items-center gap-2">
              <Button variant="outline" size="sm"><Plus className="w-4 h-4 mr-1" /> Save as labeled example</Button>
              <Button variant="outline" size="sm"><TimerReset className="w-4 h-4 mr-1" /> Re-run</Button>
            </div>
          </div>
        </footer>
      </div>
    </TooltipProvider>
  );
}

function Detail({ label, value }: { label: string; value?: string }) {
  return (
    <div className="text-xs">
      <div className="text-gray-500">{label}</div>
      <div className="font-medium break-all">{value || "—"}</div>
    </div>
  );
}

function ToggleChip({ label, active, onToggle }: { label: string; active?: boolean; onToggle?: () => void }) {
  return (
    <button
      onClick={onToggle}
      className={`text-xs px-2 py-1 rounded-full border ${active ? "bg-gray-900 text-white border-gray-900" : "bg-white hover:bg-gray-50"}`}
    >
      {label}
    </button>
  );
}

function DateField({ label, value, onChange }: { label: string; value?: Date | null; onChange: (d: Date | null) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <Label>{label}</Label>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button variant="outline" className="w-full justify-between mt-1">
            <span className="truncate text-left">
              {value ? value.toLocaleDateString() : "Select date"}
            </span>
            <ChevronDown className="w-4 h-4" />
          </Button>
        </PopoverTrigger>
        <PopoverContent align="start" className="p-0">
          <Calendar mode="single" selected={value ?? undefined} onSelect={d => { onChange(d ?? null); setOpen(false); }} />
        </PopoverContent>
      </Popover>
    </div>
  );
}
