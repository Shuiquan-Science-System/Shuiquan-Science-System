#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VoidScout V7 Patched - English edition (single-file)
- Fixes syntax and concurrency issues
- Replaces hard bidirectional verification with fusion (1 + (-1) = 0)
- Relaxes phase and quantization filtering (keeps them but softens removal)
- Uses BM25 if available and boosts exact matches
- Adds fusion floor and soft voting to avoid zeroed scores
- Keeps anchors, pledge, plugins, prefetch, autonomous agents, optional FastAPI
Save as voidscout_v7_patched_en.py
Dependencies: numpy, scikit-learn. Optional: annoy, faiss, rank_bm25, redis, prometheus_client, fastapi
"""
from __future__ import annotations
import os
import sys
import json
import time
import math
import sqlite3
import logging
import threading
import heapq
import hashlib
import argparse
import socket
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict, Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Optional libraries detection
try:
    from annoy import AnnoyIndex
    HAS_ANNOY = True
except Exception:
    HAS_ANNOY = False

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

try:
    import redis
    HAS_REDIS = True
except Exception:
    HAS_REDIS = False

try:
    from prometheus_client import CollectorRegistry, Gauge, Counter as PromCounter, start_http_server
    HAS_PROM = True
except Exception:
    HAS_PROM = False

try:
    from fastapi import FastAPI, HTTPException, Request
    from pydantic import BaseModel
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False

# -------------------------
# Basic configuration and runtime
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("voidscout_v7_patched_en")
VERSION = "VoidScout V7 Patched (EN)"

RUNTIME_CONFIG = {
    "tfidf_max_features": 8192,
    "tfidf_ngram_range": (1, 2),
    "annoy_trees": 20,
    "use_annoy": HAS_ANNOY,
    "use_faiss": HAS_FAISS,
    "use_bm25": HAS_BM25,
    "result_cache_capacity": 8192,
    "vector_cache_capacity": 8192,
    "available_drinks": ["boost", "cloak", "clone", "quantum", "autonomous"],
    "prefetcher_n": 2,
    "anchor_k": 32,
    "anchor_local_index_k": 128,
    "pledge_top_k": 64,
    "pledge_quota_per_min": 10,
    "exclusion_threshold": 0.6,
    "exclusion_phase_dim": 48,
    "enable_bidirectional_fusion": True,
    "bidirectional_fusion_weight_max": 0.6,
    "bidirectional_fusion_weight_avg": 0.4,
    "enable_multi_seed_consensus": True,
    "multi_seed_count": 3,
    "consensus_min_agree": 2,
    "enable_ratio_quantization": True,
    "ratio_quant_bins": 24,
    "ratio_split": (0.7, 0.3),
    "prometheus_port": 8001,
    "redis_url": os.environ.get("REDIS_URL", None),
    "port_range_start": 10000,
    "port_range_end": 20000,
    # Debug and relaxation flags
    "debug_print_top20": False,
    "phase_relax_prob": 0.15,
    "quantize_relax_multiplier": 0.6,
}

DB_PATH = "voidscout_v7_patched_en.db"
DOCS_JSONL = "docs.jsonl"
EMB_NPY = "embeddings.npy"
EMB_IDS = "emb_ids.json"
INDEX_BUILD_LOCK = threading.Lock()
INDEX_BUILD_THREAD: Optional[threading.Thread] = None
INDEX_READY = threading.Event()
MAX_QUERY_LEN = 1024
DEMO_LOG = "demo_output_v7_patched_en.log"

# -------------------------
# State and metrics
# -------------------------
INDEX_STATE = "unknown"
BUILD_START_TS = None
BUILD_END_TS = None
LAST_BUILD_ERROR = None
INDEX_VERSION_HASH = None
CONFIG_HASH = None
FINGERPRINT = None

METRICS = {"queries": 0, "avg_latency_ms": 0.0, "feedbacks": 0, "cache_hits": 0, "cache_requests": 0}
FEEDBACK_STORE: List[Dict[str, Any]] = []

if HAS_PROM:
    PROM_REG = CollectorRegistry()
    PROM_QPS = PromCounter("voidscout_queries_total", "Total queries", registry=PROM_REG)
    PROM_LATENCY = Gauge("voidscout_latency_ms", "Average latency ms", registry=PROM_REG)
    PROM_CACHE_HIT = PromCounter("voidscout_cache_hits_total", "Cache hits", registry=PROM_REG)
    PROM_PLEDGES = PromCounter("voidscout_pledges_total", "Pledges triggered", registry=PROM_REG)

REDIS_CLIENT = None
if HAS_REDIS and RUNTIME_CONFIG.get("redis_url"):
    try:
        REDIS_CLIENT = redis.from_url(RUNTIME_CONFIG["redis_url"])
        log.info("connected to redis")
    except Exception:
        REDIS_CLIENT = None

# -------------------------
# LRU caches
# -------------------------
class LRUCache:
    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.data = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.data:
                val = self.data.pop(key)
                self.data[key] = val
                return val
            return None

    def set(self, key, value):
        with self.lock:
            if key in self.data:
                self.data.pop(key)
            self.data[key] = value
            if len(self.data) > self.capacity:
                self.data.popitem(last=False)

    def stats(self):
        with self.lock:
            return {"size": len(self.data), "capacity": self.capacity}

RESULT_CACHE = LRUCache(RUNTIME_CONFIG["result_cache_capacity"])
VECTOR_CACHE = LRUCache(RUNTIME_CONFIG["vector_cache_capacity"])

# -------------------------
# Priority worker for background tasks
# -------------------------
class PriorityWorker:
    def __init__(self):
        self.heap = []
        self.cv = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, priority:int, func, *args, **kwargs):
        with self.cv:
            heapq.heappush(self.heap, (priority, time.time(), func, args, kwargs))
            self.cv.notify()

    def _run(self):
        while self.running:
            with self.cv:
                while not self.heap and self.running:
                    self.cv.wait(timeout=1.0)
                if not self.running:
                    break
                _, _, func, args, kwargs = heapq.heappop(self.heap)
            try:
                func(*args, **kwargs)
            except Exception:
                log.exception("priority task error")

    def stop(self):
        self.running = False
        with self.cv:
            self.cv.notify()
        self.thread.join(timeout=1.0)

PRIORITY_WORKER = PriorityWorker()

# -------------------------
# Predictive prefetcher
# -------------------------
class PredictivePrefetcher:
    def __init__(self, n=2):
        self.n = n
        self.model = {}
        self.lock = threading.Lock()

    def observe(self, query: str):
        toks = tuple(query.split())
        with self.lock:
            for i in range(len(toks)):
                prefix = tuple(toks[max(0, i-self.n+1):i+1])
                nxt = toks[i] if i < len(toks) else None
                self.model.setdefault(prefix, Counter())
                if nxt:
                    self.model[prefix][nxt] += 1

    def predict(self, query: str, top_k=3) -> List[str]:
        toks = tuple(query.split())
        prefix = toks[-(self.n-1):] if len(toks) >= (self.n-1) else toks
        prefix = tuple(prefix)
        with self.lock:
            if prefix in self.model:
                return [t for t,_ in self.model[prefix].most_common(top_k)]
        return []

PREFETCHER = PredictivePrefetcher(RUNTIME_CONFIG["prefetcher_n"])

# -------------------------
# Embedded sample documents
# -------------------------
EMBEDDED_DOCS = [
    {"id":"g001","text":"Wolff's automatic sheep-catcher: radar sniffers that aggregate faint semantic signals into high-priority targets.","source":"gray_lab","version":"v1","tags":["tracking","radar"]},
    {"id":"g002","text":"Slow Sheep's restoration scroll: records historical versions and supports rollback to any point.","source":"slow_lab","version":"v1","tags":["provenance","rollback"]},
    {"id":"g003","text":"Happy Sheep's super drink - Tracker: temporarily boosts retrieval weight for 'tracking/locating' documents.","source":"happy_lab","version":"v1","tags":["tracking","boost"]},
    {"id":"g004","text":"Hapi family's invisibility drink: example blacklist rules for filtering sensitive or noisy content.","source":"happy_lab","version":"v1","tags":["filter","safety"]},
    {"id":"g005","text":"Doppelball simulator: generates parallel candidate behavior descriptions for multi-agent retrieval fusion.","source":"cartoon_ep5","version":"v1","tags":["agent","parallel"]},
    {"id":"g006","text":"Power magnet device: aggregates related resources and signals, useful for recall enhancement and semantic clustering.","source":"cartoon_ep1","version":"v1","tags":["recall","aggregation"]},
    {"id":"g007","text":"Brain activator prototype: example synonyms and phrase expansions for query expansion.","source":"cartoon_ep4","version":"v1","tags":["expansion","genir"]},
    {"id":"g008","text":"Super drink - Accelerator: temporarily raises retriever priority to simulate session-level boost.","source":"happy_lab","version":"v1","tags":["session","boost"]},
    {"id":"g009","text":"Invisibility cloak evasion tactics: documents with noise and adversarial phrases for robustness testing.","source":"cartoon_ep3","version":"v2","tags":["robustness","adversarial"]},
    {"id":"g010","text":"Lab encyclopedia scroll: long-form example with multiple sections and version annotations for evidence tracing.","source":"slow_lab","version":"v3","tags":["longform","evidence"]},
    {"id":"g011","text":"Tracking tools comparison: lists differences among tracking/locating tools for classification and rerank rule validation.","source":"gray_lab","version":"v1","tags":["compare","tracking"]},
    {"id":"g012","text":"Energy drink collection: each drink provides a short-term strategy (invisibility, tracking, acceleration, doppelganger) for plugin combination testing.","source":"happy_lab","version":"v1","tags":["plugin","drinks"]},
]

# -------------------------
# Database helpers
# -------------------------
def ensure_docs_jsonl():
    if not os.path.exists(DOCS_JSONL):
        with open(DOCS_JSONL, "w", encoding="utf-8") as f:
            for d in EMBEDDED_DOCS:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        log.info("wrote embedded docs to %s", DOCS_JSONL)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, text TEXT, source TEXT, version TEXT, tags TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS feedback (ts INTEGER, query TEXT, doc_id TEXT, clicked INTEGER)")
    cur.execute("CREATE TABLE IF NOT EXISTS index_builds (id INTEGER PRIMARY KEY AUTOINCREMENT, start_ts INTEGER, end_ts INTEGER, status TEXT, version_hash TEXT, config_hash TEXT, fingerprint TEXT, docs_count INTEGER, log_snippet TEXT)")
    conn.commit()
    conn.close()

def import_jsonl_to_db(path: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                cur.execute("INSERT OR REPLACE INTO docs (id,text,source,version,tags) VALUES (?,?,?,?,?)",
                            (obj.get("id"), obj.get("text",""), obj.get("source",""), obj.get("version","v1"), json.dumps(obj.get("tags",[]), ensure_ascii=False)))
                count += 1
            except Exception:
                continue
    conn.commit()
    conn.close()
    log.info("imported %d docs", count)

def load_docs_from_db() -> List[Dict[str,Any]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id,text,source,version,tags FROM docs")
    rows = cur.fetchall()
    conn.close()
    docs = []
    for r in rows:
        try:
            tags = json.loads(r[4]) if r[4] else []
        except Exception:
            tags = []
        docs.append({"id": r[0], "text": r[1], "source": r[2], "version": r[3], "tags": tags})
    return docs

# -------------------------
# Hash and fingerprint helpers
# -------------------------
def compute_index_hash(docs: List[Dict[str,Any]]) -> str:
    m = hashlib.sha256()
    for d in sorted(docs, key=lambda x: x.get("id","")):
        m.update((d.get("id","") + "|" + d.get("text","")).encode("utf-8"))
    return m.hexdigest()

def compute_config_hash(config: Dict[str,Any]) -> str:
    def normalize(obj):
        if isinstance(obj, dict):
            return {k: normalize(obj[k]) for k in sorted(obj.keys())}
        if isinstance(obj, (list, tuple)):
            return [normalize(x) for x in obj]
        if isinstance(obj, bool):
            return str(obj).lower()
        return obj
    b = json.dumps(normalize(config), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

def compute_fingerprint(index_hash: str, build_start_ts: Optional[float], config_hash: str) -> str:
    ts_part = str(int(build_start_ts)) if build_start_ts else "0"
    return hashlib.sha256((index_hash + "|" + ts_part + "|" + config_hash).encode("utf-8")).hexdigest()

# -------------------------
# Retrieval globals
# -------------------------
DOCS: List[Dict[str,Any]] = []
DOC_TEXTS: List[str] = []
DOC_IDS: List[str] = []
DOC_META: Dict[str, Dict[str,Any]] = {}
tfidf_vectorizer = None
tfidf_matrix = None
ann_index = None
ann_dim = 0
bm25 = None
offline_embs = None
offline_emb_ids = None
offline_ann = None
offline_emb_dim = 0
faiss_index = None
FAISS_ID_MAP: List[int] = []

# -------------------------
# Anchors and A.T. helpers
# -------------------------
ANCHORS: List[np.ndarray] = []
ANCHOR_TO_DOCIDX: Dict[int, List[int]] = {}
ANCHOR_LOCAL_INDEX: Dict[int, List[int]] = {}
ANCHOR_SIGNATURE_MAP: Dict[str, int] = {}
ANCHOR_NEIGHBORS: Dict[int, List[int]] = {}

def compute_phase_vector(vec: np.ndarray, dim: int = RUNTIME_CONFIG["exclusion_phase_dim"]) -> np.ndarray:
    h = hashlib.sha256(vec.tobytes()).digest()
    seed = int.from_bytes(h[:8], "little") & 0xffffffff
    rng = np.random.RandomState(seed)
    proj = rng.normal(size=(dim, vec.shape[0])).astype(np.float32)
    phase = proj.dot(vec)
    n = np.linalg.norm(phase)
    return phase / n if n > 1e-12 else phase

def phase_interference(q_phase: np.ndarray, c_phase: np.ndarray) -> float:
    sim = float(np.dot(q_phase, c_phase))
    sim = max(min(sim, 1.0), -1.0)
    return 1.0 - ((sim + 1.0) / 2.0)

def ratio_quantize_vector(vec: np.ndarray) -> Tuple[Tuple[int,...], Tuple[int,...]]:
    if not RUNTIME_CONFIG["enable_ratio_quantization"]:
        return (), ()
    total_dim = vec.shape[0]
    r1 = int(total_dim * RUNTIME_CONFIG["ratio_split"][0])
    bins = RUNTIME_CONFIG["ratio_quant_bins"]
    v = vec.copy()
    n = np.linalg.norm(v)
    if n > 1e-12:
        v = v / n
    part1 = v[:r1] if r1 > 0 else np.array([], dtype=np.float32)
    part2 = v[r1:] if r1 < total_dim else np.array([], dtype=np.float32)
    def quantize_part(p):
        if p.size == 0:
            return tuple()
        clipped = np.clip(p, -1.0, 1.0)
        codes = ((clipped + 1.0) * 0.5 * (bins - 1)).astype(int)
        return tuple(int(x) for x in codes.tolist())
    return quantize_part(part1), quantize_part(part2)

def quantized_match(q_codes: Tuple[Tuple[int,...], Tuple[int,...]], c_codes: Tuple[Tuple[int,...], Tuple[int,...]]) -> int:
    if not q_codes or not c_codes:
        return 0
    p1q, p2q = q_codes
    p1c, p2c = c_codes
    matches = 0
    for a, b in zip(p1q, p1c):
        if a == b:
            matches += 1
    for a, b in zip(p2q, p2c):
        if a == b:
            matches += 1
    return matches

# -------------------------
# Reranker and plugin "drinks"
# -------------------------
class SuperReranker:
    def __init__(self):
        self.k2t = {"tracking":"tracking","locate":"tracking","restore":"provenance","invisible":"cloak","boost":"boost"}
    def update_session(self, s, q):
        hist = s.get("recent_queries", [])
        hist.append(q)
        s["recent_queries"] = hist[-200:]
    def apply(self, results, s):
        scores = {}
        for q in s.get("recent_queries", []):
            for k, t in self.k2t.items():
                if k in q:
                    scores[t] = scores.get(t, 0.0) + 1.0
        if not scores:
            return results
        m = max(scores.values())
        for k in list(scores.keys()):
            scores[k] = scores[k] / m
        out = []
        for r in results:
            mult = 1.0
            for t in r.get("tags", []):
                ts = scores.get(t, 0.0)
                if ts >= 0.6:
                    mult *= 1.5
                elif 0 < ts <= 0.2:
                    mult *= 0.8
            r["score"] = round(float(r["score"]) * mult, 6)
            out.append(r)
        return sorted(out, key=lambda x: x["score"], reverse=True)

reranker = SuperReranker()

class Drink:
    name = "base"
    def transform_query(self, q, s): return q
    def session_boosts(self, s): return {}
    def filter_candidates(self, cands, s): return cands
    def parallel_variants(self, q, s): return []

class Boost(Drink):
    name = "boost"
    def __init__(self, terms=["enhance","accelerate"], val=1.5, ttl=120):
        self.terms = terms
        self.val = val
        self.ttl = ttl
    def session_boosts(self, s):
        now = time.time()
        key = f"boost_{self.name}_start"
        start = s.get(key)
        if start is None:
            s[key] = now
            start = now
        if now - start <= self.ttl:
            return {t: self.val for t in self.terms}
        return {}
    def transform_query(self, q, s):
        return q + " " + " ".join(self.terms)

class Cloak(Drink):
    name = "cloak"
    def __init__(self, blacklist=None):
        self.blacklist = blacklist or []
    def filter_candidates(self, cands, s):
        bl = [b.lower() for b in self.blacklist]
        out = []
        for c in cands:
            text = c.get("text","").lower()
            if any(b in text for b in bl):
                continue
            out.append(c)
        return out

class Clone(Drink):
    name = "clone"
    def parallel_variants(self, q, s):
        return [{"query": q}, {"query": q + " plan"}, {"query": q + " invention"}]

class Quantum(Drink):
    name = "quantum"
    def __init__(self, prefetch_k=8, priority=0):
        self.prefetch_k = prefetch_k
        self.priority = priority
    def parallel_variants(self, q, s):
        return [{"query": q}, {"query": q + " tracking"}, {"query": q + " plan"}]
    def trigger_prefetch(self, q, s):
        PREFETCHER.observe(q)
        preds = PREFETCHER.predict(q, top_k=3)
        PRIORITY_WORKER.submit(self.priority, prefetch_candidates, q, preds, self.prefetch_k)

class Autonomous(Drink):
    name = "autonomous"
    def __init__(self, prefetch_k=12, priority=-1):
        self.prefetch_k = prefetch_k
        self.priority = priority
    def parallel_variants(self, q, s):
        return [{"query": q}, {"query": q + " plan"}]
    def trigger_autonomous_tasks(self, s):
        PRIORITY_WORKER.submit(self.priority, run_autonomous_agents, s, self.prefetch_k)

AVAILABLE_DRINKS = {
    "boost": Boost(),
    "cloak": Cloak(blacklist=["danger","forbidden"]),
    "clone": Clone(),
    "quantum": Quantum(),
    "autonomous": Autonomous()
}

def apply_drinks(query, drink_names, session_state):
    variants = [{"query": query}]
    boosts = {}
    filters = []
    for name in (drink_names or []):
        p = AVAILABLE_DRINKS.get(name)
        if not p:
            continue
        try:
            for v in p.parallel_variants(query, session_state):
                if isinstance(v, dict) and "query" in v:
                    variants.append(v)
        except Exception:
            log.exception("plugin parallel_variants error")
        try:
            query = p.transform_query(query, session_state)
        except Exception:
            log.exception("plugin transform_query error")
        try:
            b = p.session_boosts(session_state)
            if b:
                boosts.update(b)
        except Exception:
            log.exception("plugin session_boosts error")
        try:
            filters.append(p.filter_candidates)
        except Exception:
            log.exception("plugin filter_candidates error")
        try:
            if hasattr(p, "trigger_prefetch"):
                p.trigger_prefetch(query, session_state)
            if hasattr(p, "trigger_autonomous_tasks"):
                p.trigger_autonomous_tasks(session_state)
        except Exception:
            log.exception("plugin trigger error")
    seen = set()
    uniq = []
    for v in variants:
        qv = v.get("query") if isinstance(v, dict) else v
        if qv not in seen:
            seen.add(qv)
            uniq.append(v if isinstance(v, dict) else {"query": qv})
    return {"variants": uniq, "boosts": boosts, "filters": filters, "final_query": query}

# -------------------------
# Prefetch and autonomous tasks
# -------------------------
def prefetch_candidates(query: str, preds: List[str], prefetch_k: int):
    start = time.time()
    queries = [query] + (preds or [])
    for q in queries:
        try:
            q_vec = tfidf_vectorizer.transform([q]).toarray().astype(np.float32)
            q_vec = normalize(q_vec, axis=1)[0]
            VECTOR_CACHE.set(f"vec:{q}", q_vec)
            if RUNTIME_CONFIG["use_annoy"] and HAS_ANNOY and ann_index is not None:
                nn = ann_index.get_nns_by_vector(q_vec.tolist(), prefetch_k, include_distances=True)
                idxs = nn[0] if isinstance(nn[0], list) else nn[0]
            elif RUNTIME_CONFIG["use_faiss"] and HAS_FAISS and faiss_index is not None:
                D, I = faiss_index.search(np.expand_dims(q_vec, axis=0), prefetch_k)
                idxs = [int(i) for i in I[0] if i >= 0]
            else:
                sims = cosine_similarity(q_vec.reshape(1,-1), tfidf_matrix)[0]
                idxs = list(np.argsort(-sims)[:prefetch_k])
            cand_list = []
            for idx in idxs:
                doc_id = DOC_IDS[idx]
                cand_list.append({"id": doc_id, "text": DOC_TEXTS[idx], "source": DOC_META[doc_id]["source"], "version": DOC_META[doc_id]["version"], "score": 0.0})
            RESULT_CACHE.set(f"prefetch:{q}", cand_list)
            if REDIS_CLIENT:
                try:
                    REDIS_CLIENT.setex(f"prefetch:{q}", 60, json.dumps(cand_list, ensure_ascii=False))
                except Exception:
                    pass
        except Exception:
            log.exception("prefetch error")
    METRICS["prefetch_latency_ms"] = ((METRICS.get("prefetch_latency_ms", 0.0) * 0.9) + ((time.time() - start) * 1000.0) * 0.1)

def run_autonomous_agents(session_state: Dict[str,Any], prefetch_k: int):
    start = time.time()
    session_id = session_state.get("session_id", f"sess_{int(time.time())}")
    queries = session_state.get("recent_queries", [])[-3:]
    merged = []
    try:
        q = queries[-1] if queries else ""
        if q:
            q_vec = tfidf_vectorizer.transform([q]).toarray().astype(np.float32)
            q_vec = normalize(q_vec, axis=1)[0]
            sims = cosine_similarity(q_vec.reshape(1,-1), tfidf_matrix)[0]
            top_idxs = np.argsort(-sims)[:prefetch_k]
            for idx in top_idxs:
                merged.append({"id": DOC_IDS[idx], "text": DOC_TEXTS[idx], "source": DOC_META[DOC_IDS[idx]]["source"], "score": float(sims[idx])})
    except Exception:
        pass
    try:
        time.sleep(0.02)
        if any(k in q for k in ["tracking","locate","catch"]):
            for d in DOCS:
                if "tracking" in d.get("tags", []):
                    merged.append({"id": d["id"], "text": d["text"], "source": d["source"], "score": 0.5})
    except Exception:
        pass
    seen = set()
    final = []
    for c in merged:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        final.append(c)
    RESULT_CACHE.set(f"autonomous:{session_id}", final)
    log.info("autonomous finished for %s candidates=%d latency_ms=%.2f", session_id, len(final), (time.time() - start) * 1000.0)

# -------------------------
# Pledge mechanism
# -------------------------
PLEDGE_LOCK = threading.Lock()
PLEDGE_STATE: Dict[str, Dict[str, Any]] = {}

def pledge_allowed(session_id: str) -> bool:
    now = time.time()
    with PLEDGE_LOCK:
        s = PLEDGE_STATE.setdefault(session_id, {"window_start": now, "count": 0, "active": []})
        if now - s["window_start"] > 60:
            s["window_start"] = now
            s["count"] = 0
        if s["count"] < RUNTIME_CONFIG["pledge_quota_per_min"]:
            s["count"] += 1
            return True
        return False

def start_pledge(session_id: str, key_dims: List[int], expiry_s: int = 30):
    now = time.time()
    with PLEDGE_LOCK:
        s = PLEDGE_STATE.setdefault(session_id, {"window_start": now, "count": 0, "active": []})
        pledge = {"start": now, "expiry": now + expiry_s, "key_dims": key_dims}
        s["active"].append(pledge)
        return pledge

def end_expired_pledges():
    now = time.time()
    with PLEDGE_LOCK:
        for sid, s in list(PLEDGE_STATE.items()):
            s["active"] = [p for p in s["active"] if p["expiry"] > now]

def select_key_dimensions_by_query(query: str, top_k: int = 128) -> List[int]:
    try:
        toks = query.split()
        tok_scores = Counter(toks)
        top_tokens = [t for t,_ in tok_scores.most_common(20)]
        dims = []
        if tfidf_vectorizer is not None:
            vocab = tfidf_vectorizer.vocabulary_
            for t in top_tokens:
                if t in vocab:
                    dims.append(vocab[t])
                    if len(dims) >= top_k:
                        break
        if tfidf_matrix is not None and len(dims) < top_k:
            total = tfidf_matrix.shape[1]
            step = max(1, total // (top_k - len(dims) + 1))
            for i in range(0, total, step):
                if i not in dims:
                    dims.append(i)
                if len(dims) >= top_k:
                    break
        return dims[:top_k]
    except Exception:
        if tfidf_matrix is not None:
            return list(range(min(top_k, tfidf_matrix.shape[1])))
        return []

# -------------------------
# Anchors and FAISS helpers
# -------------------------
def build_anchors(k: int = RUNTIME_CONFIG["anchor_k"], local_k: int = RUNTIME_CONFIG["anchor_local_index_k"]):
    global ANCHORS, ANCHOR_TO_DOCIDX, ANCHOR_LOCAL_INDEX, ANCHOR_SIGNATURE_MAP, ANCHOR_NEIGHBORS
    if tfidf_matrix is None or len(DOC_TEXTS) == 0:
        return
    n_docs = len(DOC_TEXTS)
    rng = np.random.RandomState(42)
    seeds_idx = rng.choice(n_docs, size=min(k, n_docs), replace=False)
    anchors = [tfidf_matrix[i] for i in seeds_idx]
    anchors = np.vstack(anchors)
    sims = cosine_similarity(anchors, tfidf_matrix)
    assign = np.argmax(sims, axis=0)
    ANCHORS = []
    ANCHOR_TO_DOCIDX = {}
    ANCHOR_LOCAL_INDEX = {}
    for a_idx in range(anchors.shape[0]):
        docidxs = [i for i, v in enumerate(assign) if v == a_idx]
        ANCHOR_TO_DOCIDX[a_idx] = docidxs
        if len(docidxs) == 0:
            ANCHOR_LOCAL_INDEX[a_idx] = []
            ANCHORS.append(anchors[a_idx])
            continue
        anchor_vec = np.mean(tfidf_matrix[docidxs], axis=0)
        ANCHORS.append(anchor_vec)
        simslocal = cosine_similarity(anchor_vec.reshape(1,-1), tfidf_matrix)[0]
        ANCHOR_LOCAL_INDEX[a_idx] = list(np.argsort(-simslocal)[:min(local_k, len(simslocal))])
    ANCHOR_SIGNATURE_MAP = {}
    for a_idx, anchor in enumerate(ANCHORS):
        ANCHOR_SIGNATURE_MAP[anchor_signature(anchor)] = a_idx
    ANCHOR_NEIGHBORS = {}
    if len(ANCHORS) > 1:
        anchor_mat = np.vstack(ANCHORS)
        a_sims = cosine_similarity(anchor_mat, anchor_mat)
        for i in range(a_sims.shape[0]):
            neigh = list(np.argsort(-a_sims[i])[:5])
            ANCHOR_NEIGHBORS[i] = [n for n in neigh if n != i]
    log.info("anchors built: anchors=%d", len(ANCHORS))

def anchor_signature(vec: np.ndarray, bits: int = 16) -> str:
    seed = int(hashlib.sha256(b"anchor_sig_seed").hexdigest()[:8], 16) & 0xffffffff
    rng = np.random.RandomState(seed)
    proj = rng.normal(size=(bits, vec.shape[0])).astype(np.float32)
    signs = (proj.dot(vec) > 0).astype(int)
    s = "".join(str(b) for b in signs.tolist())
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def anchor_hop_search(query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
    sig = anchor_signature(query_vec)
    anchor_id = ANCHOR_SIGNATURE_MAP.get(sig, None)
    candidates = []
    if anchor_id is not None:
        for idx in ANCHOR_LOCAL_INDEX.get(anchor_id, []):
            candidates.append((idx, float(np.dot(query_vec, tfidf_matrix[idx]))))
        if len(candidates) < top_k:
            for n in ANCHOR_NEIGHBORS.get(anchor_id, [])[:3]:
                for idx in ANCHOR_LOCAL_INDEX.get(n, [])[:top_k]:
                    candidates.append((idx, float(np.dot(query_vec, tfidf_matrix[idx]))))
    else:
        if len(ANCHORS) > 0:
            anchor_mat = np.vstack(ANCHORS)
            sims = cosine_similarity(query_vec.reshape(1,-1), anchor_mat)[0]
            best = int(np.argmax(sims))
            for idx in ANCHOR_LOCAL_INDEX.get(best, [])[:top_k*2]:
                candidates.append((idx, float(np.dot(query_vec, tfidf_matrix[idx]))))
    uniq = {}
    for idx, sim in candidates:
        if idx not in uniq or sim > uniq[idx]:
            uniq[idx] = sim
    out = sorted(uniq.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return out

def build_faiss_index():
    global faiss_index, FAISS_ID_MAP
    if not HAS_FAISS or tfidf_matrix is None:
        return
    d = tfidf_matrix.shape[1]
    xb = tfidf_matrix.astype(np.float32)
    try:
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(xb)
        index.add(xb)
        faiss_index = index
        FAISS_ID_MAP = list(range(len(DOC_TEXTS)))
        log.info("FAISS index built")
    except Exception:
        log.exception("faiss build failed")
        faiss_index = None

# -------------------------
# Index build background worker
# -------------------------
def build_indices_background(annoy_trees:int=RUNTIME_CONFIG["annoy_trees"]):
    global DOCS, DOC_TEXTS, DOC_IDS, DOC_META, tfidf_vectorizer, tfidf_matrix, ann_index, ann_dim, bm25, offline_embs, offline_ann, offline_emb_ids, offline_emb_dim, faiss_index
    global INDEX_STATE, BUILD_START_TS, BUILD_END_TS, LAST_BUILD_ERROR, INDEX_VERSION_HASH, CONFIG_HASH, FINGERPRINT
    with INDEX_BUILD_LOCK:
        INDEX_READY.clear()
        INDEX_STATE = "building"
        BUILD_START_TS = time.time()
        BUILD_END_TS = None
        LAST_BUILD_ERROR = None
        log.info("background index build started")
        audit_conn = sqlite3.connect(DB_PATH)
        audit_cur = audit_conn.cursor()
        audit_cur.execute("INSERT INTO index_builds (start_ts, status) VALUES (?,?)", (int(BUILD_START_TS), "building"))
        build_row_id = audit_cur.lastrowid
        audit_conn.commit()
        try:
            docs = load_docs_from_db()
            if not docs:
                docs = EMBEDDED_DOCS
                import_jsonl_to_db(DOCS_JSONL)
                docs = load_docs_from_db()
            INDEX_VERSION_HASH = compute_index_hash(docs)
            CONFIG_HASH = compute_config_hash(RUNTIME_CONFIG)
            FINGERPRINT = compute_fingerprint(INDEX_VERSION_HASH, BUILD_START_TS, CONFIG_HASH)
            DOCS = docs.copy()
            DOC_TEXTS = [d["text"] for d in DOCS]
            DOC_IDS = [d["id"] for d in DOCS]
            DOC_META = {d["id"]: {"source": d.get("source","unknown"), "version": d.get("version","v1"), "tags": d.get("tags",[])} for d in DOCS}
            log.info("building TF-IDF matrix for %d docs...", len(DOC_TEXTS))
            tfidf_vectorizer = TfidfVectorizer(max_features=RUNTIME_CONFIG["tfidf_max_features"], ngram_range=RUNTIME_CONFIG["tfidf_ngram_range"])
            tfidf_matrix = tfidf_vectorizer.fit_transform(DOC_TEXTS).toarray().astype(np.float32)
            tfidf_matrix = normalize(tfidf_matrix, axis=1)
            ann_dim = tfidf_matrix.shape[1]
            log.info("TF-IDF built dim=%d", ann_dim)
            if RUNTIME_CONFIG["use_annoy"] and HAS_ANNOY:
                try:
                    ann_index = AnnoyIndex(ann_dim, metric='angular')
                    for i in range(len(DOC_TEXTS)):
                        ann_index.add_item(i, tfidf_matrix[i].tolist())
                    ann_index.build(annoy_trees)
                    log.info("Annoy built")
                except Exception:
                    log.exception("Annoy build failed")
                    ann_index = None
            else:
                ann_index = None
            if RUNTIME_CONFIG["use_faiss"] and HAS_FAISS:
                try:
                    build_faiss_index()
                except Exception:
                    log.exception("FAISS build failed")
                    faiss_index = None
            if RUNTIME_CONFIG["use_bm25"] and HAS_BM25:
                try:
                    tokenized = [text.split() for text in DOC_TEXTS]
                    bm25 = BM25Okapi(tokenized)
                    log.info("BM25 built")
                except Exception:
                    log.exception("BM25 build failed")
                    bm25 = None
            else:
                bm25 = None
            # offline embeddings load (split statements to avoid syntax issues)
            if os.path.exists(EMB_NPY) and os.path.exists(EMB_IDS):
                try:
                    emb = np.load(EMB_NPY)
                    with open(EMB_IDS, "r", encoding="utf-8") as f:
                        ids = json.load(f)
                    if emb.ndim == 2 and len(ids) == emb.shape[0]:
                        offline_embs = emb.astype(np.float32)
                        offline_emb_ids = ids
                        offline_emb_dim = offline_embs.shape[1]
                        if HAS_ANNOY:
                            offline_ann = AnnoyIndex(offline_emb_dim, metric='angular')
                            for i in range(len(offline_embs)):
                                offline_ann.add_item(i, offline_embs[i].tolist())
                            offline_ann.build(10)
                        log.info("offline embeddings loaded shape=%s", offline_embs.shape)
                except Exception:
                    log.exception("failed to load offline embeddings")
            try:
                build_anchors(k=RUNTIME_CONFIG["anchor_k"], local_k=RUNTIME_CONFIG["anchor_local_index_k"])
            except Exception:
                log.exception("anchor build failed")
            BUILD_END_TS = time.time()
            INDEX_STATE = "ready"
            INDEX_READY.set()
            audit_cur.execute("UPDATE index_builds SET end_ts=?, status=?, version_hash=?, config_hash=?, fingerprint=?, docs_count=?, log_snippet=? WHERE id=?",
                              (int(BUILD_END_TS), "ready", INDEX_VERSION_HASH, CONFIG_HASH, FINGERPRINT, len(DOC_TEXTS), "ok", build_row_id))
            audit_conn.commit()
            audit_conn.close()
            log.info("background index build finished: docs=%d", len(DOC_TEXTS))
        except Exception as e:
            BUILD_END_TS = time.time()
            LAST_BUILD_ERROR = str(e)
            INDEX_STATE = "failed"
            INDEX_READY.clear()
            try:
                audit_cur.execute("UPDATE index_builds SET end_ts=?, status=?, log_snippet=? WHERE id=?",
                                  (int(BUILD_END_TS), "failed", str(e)[:2000], build_row_id))
                audit_conn.commit()
                audit_conn.close()
            except Exception:
                pass
            log.exception("background index build encountered an error")

def start_background_index():
    global INDEX_BUILD_THREAD
    if INDEX_BUILD_THREAD and INDEX_BUILD_THREAD.is_alive():
        return
    INDEX_BUILD_THREAD = threading.Thread(target=build_indices_background, args=(RUNTIME_CONFIG["annoy_trees"],), daemon=True)
    INDEX_BUILD_THREAD.start()
    log.info("index build thread launched")

# -------------------------
# Bidirectional fusion (1 + (-1) = 0 "puzzle fusion")
# -------------------------
def bidirectional_fuse(original_query: str, doc_id: str) -> Tuple[float, Dict[str, float]]:
    """
    Fusion strategy:
    - compute forward_sim (query -> doc) and reverse_sim (doc -> best matching doc similarity)
    - combine max(forward, reverse) and average(forward, reverse) with configurable weights
    - return fused confidence and component values
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT text FROM docs WHERE id=?", (doc_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return 0.0, {"forward": 0.0, "reverse": 0.0}
        doc_text = row[0]
        q_vec = tfidf_vectorizer.transform([original_query]).toarray().astype(np.float32)
        q_vec = normalize(q_vec, axis=1)[0]
        doc_vec = tfidf_vectorizer.transform([doc_text]).toarray().astype(np.float32)
        doc_vec = normalize(doc_vec, axis=1)[0]
        forward_sim = float(np.dot(q_vec, doc_vec))
        sims = cosine_similarity(doc_vec.reshape(1,-1), tfidf_matrix)[0]
        best_idx_for_doc = int(np.argmax(sims))
        reverse_sim = float(sims[best_idx_for_doc])
        w_max = RUNTIME_CONFIG.get("bidirectional_fusion_weight_max", 0.6)
        w_avg = RUNTIME_CONFIG.get("bidirectional_fusion_weight_avg", 0.4)
        fused = (w_max * max(forward_sim, reverse_sim)) + (w_avg * ((forward_sim + reverse_sim) / 2.0))
        fused = max(0.0, min(1.0, fused))
        return fused, {"forward": round(forward_sim, 6), "reverse": round(reverse_sim, 6)}
    except Exception:
        log.exception("bidirectional_fuse error")
        return 0.0, {"forward": 0.0, "reverse": 0.0}

# -------------------------
# Multi-seed consensus wrapper
# -------------------------
def multi_seed_search_wrapper(query: str, top_k: int, drinks: List[str], session_state: Dict[str,Any], client_ip: Optional[str]) -> Dict[str,Any]:
    seeds = RUNTIME_CONFIG["multi_seed_count"] if RUNTIME_CONFIG["enable_multi_seed_consensus"] else 1
    seed_results = []
    for s in range(seeds):
        q_variant = query if s == 0 else query + (" _s" + str(s))
        res = search_pipeline_core(q_variant, top_k=top_k, drinks=drinks, session_state=session_state, client_ip=client_ip, allow_multi_seed=False)
        seed_results.append(res)
    if seeds == 1:
        return seed_results[0]
    counts = Counter()
    for res in seed_results:
        ids = [r["id"] for r in res.get("results", [])]
        counts.update(ids)
    agree = [doc for doc, c in counts.items() if c >= RUNTIME_CONFIG["consensus_min_agree"]]
    score_map = defaultdict(list)
    for res in seed_results:
        for r in res.get("results", []):
            if r["id"] in agree:
                score_map[r["id"]].append(r["score"])
    final_list = []
    for doc_id, scores in score_map.items():
        avg_score = sum(scores) / len(scores)
        try:
            idx = DOC_IDS.index(doc_id)
            final_list.append({"id": doc_id, "text": DOC_TEXTS[idx], "source": DOC_META[doc_id]["source"], "version": DOC_META[doc_id]["version"], "tags": DOC_META[doc_id].get("tags", []), "score": round(float(avg_score), 6)})
        except Exception:
            continue
    final_list = sorted(final_list, key=lambda x: x["score"], reverse=True)[:top_k]
    return {"query": query, "drinks": drinks, "results": final_list, "meta": {"timestamp": int(time.time()), "consensus": True}}

# -------------------------
# Core search pipeline (fusion replaces hard bidirectional filter)
# -------------------------
LAST_REQUEST_TS = {}
RATE_LIMIT_WINDOW = 0.12
ALPHA_LOCK = threading.Lock()
ALPHA = 0.6
ALPHA_STATS = {"dense_clicks": 0, "sparse_clicks": 0}

def estimate_query_cost(query: str, session_state: Dict[str,Any]) -> float:
    toks = query.split()
    length = len(toks)
    avg_token_len = sum(len(t) for t in toks) / max(1, length)
    missing = 0
    if tfidf_vectorizer is not None:
        vocab = tfidf_vectorizer.vocabulary_
        for t in toks:
            if t not in vocab:
                missing += 1
    rarity = missing / max(1, length)
    load = min(1.0, METRICS["avg_latency_ms"] / 200.0) if METRICS["avg_latency_ms"] > 0 else 0.0
    score = min(1.0, (length / 20.0) * 0.4 + (avg_token_len / 8.0) * 0.2 + rarity * 0.3 + load * 0.1)
    return score

def search_pipeline_core(query, top_k=10, alpha=None, drinks=None, session_state=None, client_ip=None, allow_multi_seed=True):
    start = time.time()
    if session_state is None:
        session_state = {}
    if drinks is None:
        drinks = []
    if alpha is None:
        with ALPHA_LOCK:
            alpha = ALPHA
    if client_ip:
        last = LAST_REQUEST_TS.get(client_ip, 0)
        if time.time() - last < RATE_LIMIT_WINDOW:
            raise Exception("rate limit")
        LAST_REQUEST_TS[client_ip] = time.time()
    METRICS["cache_requests"] += 1
    cache_key = f"result:{query}:{','.join(sorted(drinks))}:{top_k}"
    if REDIS_CLIENT:
        try:
            raw = REDIS_CLIENT.get(cache_key)
            if raw:
                res = json.loads(raw)
                METRICS["cache_hits"] += 1
                if HAS_PROM:
                    PROM_CACHE_HIT.inc()
                return res
        except Exception:
            pass
    cached = RESULT_CACHE.get(cache_key)
    if cached is not None:
        METRICS["cache_hits"] += 1
        if HAS_PROM:
            PROM_CACHE_HIT.inc()
        METRICS["queries"] += 1
        METRICS["avg_latency_ms"] = ((METRICS["avg_latency_ms"] * (METRICS["queries"] - 1)) + 1.0) / METRICS["queries"]
        return {"query": query, "drinks": drinks, "results": cached, "meta": {"alpha": alpha, "timestamp": int(time.time()), "latency_ms": 0.5, "cache_hit": True}}
    INDEX_READY.wait(timeout=60)
    try:
        reranker.update_session(session_state, query)
    except Exception:
        pass
    plan = apply_drinks(query, drinks, session_state)
    variants = plan["variants"]
    boosts = plan["boosts"]
    filters = plan["filters"]
    num_docs = len(DOC_TEXTS)
    sparse_scores = np.zeros(num_docs, dtype=np.float32)
    dense_scores = np.zeros(num_docs, dtype=np.float32)
    candidate_idxs = []
    q_vec_full = tfidf_vectorizer.transform([query]).toarray().astype(np.float32)
    q_vec_full = normalize(q_vec_full, axis=1)[0]
    q_phase = compute_phase_vector(q_vec_full)
    q_qcodes = ratio_quantize_vector(q_vec_full) if RUNTIME_CONFIG["enable_ratio_quantization"] else ((), ())
    cost = estimate_query_cost(query, session_state)
    pledge_mode = False
    pledge_dims = []
    if cost > 0.75 and pledge_allowed(session_state.get("session_id", "anon")):
        pledge_mode = True
        pledge_dims = select_key_dimensions_by_query(query, top_k=min(RUNTIME_CONFIG["pledge_top_k"], q_vec_full.shape[0]))
        start_pledge(session_state.get("session_id", "anon"), pledge_dims, expiry_s=30)
        if HAS_PROM:
            PROM_PLEDGES.inc()
    try:
        anchor_candidates = anchor_hop_search(q_vec_full, top_k=max(50, top_k * 3))
        for idx, sim in anchor_candidates:
            candidate_idxs.append(int(idx))
            dense_scores[idx] = max(dense_scores[idx], float(sim))
    except Exception:
        pass
    if len(candidate_idxs) < max(50, top_k):
        if RUNTIME_CONFIG["use_annoy"] and HAS_ANNOY and ann_index is not None:
            try:
                nn = ann_index.get_nns_by_vector(q_vec_full.tolist(), max(200, top_k * 5), include_distances=True)
                idxs = nn[0] if isinstance(nn[0], list) else nn[0]
                dists = nn[1] if len(nn) > 1 else None
                for pos, idx in enumerate(idxs):
                    candidate_idxs.append(int(idx))
                    sim = 1.0 - (dists[pos] / 2.0) if dists is not None else 0.0
                    dense_scores[idx] = max(dense_scores[idx], float(sim))
            except Exception:
                sims = cosine_similarity(q_vec_full.reshape(1,-1), tfidf_matrix)[0]
                top_idxs = np.argsort(-sims)[:max(200, top_k * 5)]
                for idx in top_idxs:
                    candidate_idxs.append(int(idx))
                    dense_scores[idx] = max(dense_scores[idx], float(sims[idx]))
        elif RUNTIME_CONFIG["use_faiss"] and HAS_FAISS and faiss_index is not None:
            try:
                qn = q_vec_full.copy().astype(np.float32)
                faiss.normalize_L2(qn.reshape(1,-1))
                D, I = faiss_index.search(np.expand_dims(qn, axis=0), max(200, top_k * 5))
                idxs = [int(i) for i in I[0] if i >= 0]
                for idx in idxs:
                    candidate_idxs.append(int(idx))
                    dense_scores[idx] = max(dense_scores[idx], float(np.dot(q_vec_full, tfidf_matrix[idx])))
            except Exception:
                sims = cosine_similarity(q_vec_full.reshape(1,-1), tfidf_matrix)[0]
                top_idxs = np.argsort(-sims)[:max(200, top_k * 5)]
                for idx in top_idxs:
                    candidate_idxs.append(int(idx))
                    dense_scores[idx] = max(dense_scores[idx], float(sims[idx]))
        else:
            sims = cosine_similarity(q_vec_full.reshape(1,-1), tfidf_matrix)[0]
            top_idxs = np.argsort(-sims)[:max(200, top_k * 5)]
            for idx in top_idxs:
                candidate_idxs.append(int(idx))
                dense_scores[idx] = max(dense_scores[idx], float(sims[idx]))
    # Debug print top20 raw sims if enabled
    if RUNTIME_CONFIG.get("debug_print_top20", False):
        sims = cosine_similarity(q_vec_full.reshape(1,-1), tfidf_matrix)[0]
        top20 = np.argsort(-sims)[:20]
        print("DEBUG top20 sims:", [(DOC_IDS[i], round(float(sims[i]),4), round(float(dense_scores[i]),4), round(float(sparse_scores[i]),4)) for i in top20])
    # Build sparse scores using BM25 if available, and apply exact-match boost
    if bm25 is not None:
        try:
            q_tokens = query.split()
            bm_raw = bm25.get_scores(q_tokens)
            bm_arr = np.array(bm_raw, dtype=np.float32)
            if bm_arr.max() - bm_arr.min() > 1e-9:
                bm_norm = (bm_arr - bm_arr.min()) / (bm_arr.max() - bm_arr.min())
            else:
                bm_norm = np.zeros_like(bm_arr)
            # Exact-match boost: if query substring in doc text, raise score
            for i, doc_text in enumerate(DOC_TEXTS):
                if query.strip() and query.strip() in doc_text:
                    bm_norm[i] = max(bm_norm[i], 0.9)
            sparse_scores = np.maximum(sparse_scores, bm_norm)
        except Exception:
            log.exception("bm25 scoring failed")
    # If BM25 not available, compute sparse via TF-IDF sims for variants
    for var in variants:
        qv = var["query"]
        if bm25 is None:
            qv_vec = tfidf_vectorizer.transform([qv]).toarray().astype(np.float32)
            qv_vec = normalize(qv_vec, axis=1)
            sims = cosine_similarity(qv_vec, tfidf_matrix)[0]
            if sims.max() - sims.min() > 1e-9:
                bm_norm = (sims - sims.min()) / (sims.max() - sims.min())
            else:
                bm_norm = np.zeros_like(sims)
            # exact-match boost for this variant
            for i, doc_text in enumerate(DOC_TEXTS):
                if qv.strip() and qv.strip() in doc_text:
                    bm_norm[i] = max(bm_norm[i], 0.9)
            sparse_scores = np.maximum(sparse_scores, bm_norm)
    final_candidates = []
    seen = set()
    for idx in candidate_idxs:
        if idx in seen:
            continue
        seen.add(idx)
        c_vec = tfidf_matrix[idx]
        c_phase = compute_phase_vector(c_vec)
        excl = phase_interference(q_phase, c_phase)
        # Relaxed phase filtering: allow strong dense candidates or small random keep
        if excl > RUNTIME_CONFIG["exclusion_threshold"]:
            if dense_scores[idx] >= 0.6:
                pass
            else:
                if np.random.rand() < RUNTIME_CONFIG.get("phase_relax_prob", 0.15):
                    pass
                else:
                    continue
        if dense_scores[idx] == 0.0:
            dense_scores[idx] = float(np.dot(q_vec_full, c_vec))
        if RUNTIME_CONFIG["enable_ratio_quantization"]:
            c_qcodes = ratio_quantize_vector(c_vec)
            qmatch = quantized_match(q_qcodes, c_qcodes)
            if qmatch == 0:
                # relax: reduce weight instead of discarding
                dense_scores[idx] *= RUNTIME_CONFIG.get("quantize_relax_multiplier", 0.6)
        final_candidates.append(idx)
    if pledge_mode and len(pledge_dims) > 0:
        q_proj = q_vec_full[pledge_dims]
        for idx in final_candidates:
            sim = float(np.dot(q_proj, tfidf_matrix[idx][pledge_dims]))
            dense_scores[idx] = max(dense_scores[idx], sim)
    results = []
    for idx in final_candidates:
        s_dense = float(dense_scores[idx])
        s_sparse = float(sparse_scores[idx])
        boost_mult = 1.0
        doc_text = DOC_TEXTS[idx].lower()
        for term, val in boosts.items():
            if term.lower() in doc_text:
                boost_mult *= val
        fused_score = (alpha * s_dense + (1.0 - alpha) * s_sparse) * boost_mult
        doc_id = DOC_IDS[idx]
        if RUNTIME_CONFIG.get("enable_bidirectional_fusion", True):
            fuse_val, comps = bidirectional_fuse(query, doc_id)
            forward = comps.get("forward", 0.0)
            reverse = comps.get("reverse", 0.0)
            soft_factor = 0.5 + 0.5 * fuse_val
            if max(forward, reverse) > 0.75:
                soft_factor = max(soft_factor, 0.9)
            fused_score = fused_score * soft_factor
        # absolute floor to avoid zeroing out
        fused_score = max(fused_score, 1e-4)
        results.append({"id": doc_id, "text": DOC_TEXTS[idx], "source": DOC_META[doc_id]["source"], "version": DOC_META[doc_id]["version"], "tags": DOC_META[doc_id].get("tags", []), "dense_score": round(s_dense, 6), "sparse_score": round(s_sparse, 6), "score": round(float(fused_score), 6)})
    for f in filters:
        try:
            results = f(results, session_state)
        except Exception:
            pass
    results = reranker.apply(results, session_state)
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    RESULT_CACHE.set(cache_key, results)
    if REDIS_CLIENT:
        try:
            REDIS_CLIENT.setex(cache_key, 60, json.dumps(results, ensure_ascii=False))
        except Exception:
            pass
    latency = (time.time() - start) * 1000.0
    METRICS["queries"] += 1
    METRICS["avg_latency_ms"] = ((METRICS["avg_latency_ms"] * (METRICS["queries"] - 1)) + latency) / METRICS["queries"]
    if HAS_PROM:
        try:
            PROM_QPS.inc()
            PROM_LATENCY.set(METRICS["avg_latency_ms"])
        except Exception:
            pass
    end_expired_pledges()
    return {"query": query, "drinks": drinks, "results": results, "meta": {"alpha": alpha, "timestamp": int(time.time()), "latency_ms": round(latency, 2), "cache_hit": False}}

def search_pipeline(query, top_k=10, alpha=None, drinks=None, session_state=None, client_ip=None):
    if RUNTIME_CONFIG["enable_multi_seed_consensus"]:
        return multi_seed_search_wrapper(query, top_k, drinks or [], session_state or {}, client_ip)
    else:
        return search_pipeline_core(query, top_k=top_k, alpha=alpha, drinks=drinks, session_state=session_state, client_ip=client_ip)

# -------------------------
# Feedback and adaptive alpha
# -------------------------
def record_feedback(query, doc_id, clicked, session_state=None):
    global ALPHA, ALPHA_STATS
    METRICS["feedbacks"] += 1
    FEEDBACK_STORE.append({"ts": int(time.time()), "query": query, "doc_id": doc_id, "clicked": int(bool(clicked))})
    if clicked:
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT text FROM docs WHERE id=?", (doc_id,))
            row = cur.fetchone()
            conn.close()
            if row:
                text = row[0]
                if any(tok in text for tok in query.split()):
                    ALPHA_STATS["sparse_clicks"] += 1
                else:
                    ALPHA_STATS["dense_clicks"] += 1
        except Exception:
            pass
    total = ALPHA_STATS["dense_clicks"] + ALPHA_STATS["sparse_clicks"]
    if total >= 200:
        with ALPHA_LOCK:
            dense = ALPHA_STATS["dense_clicks"] / max(1, total)
            ALPHA = 0.3 + 0.6 * dense
            log.info("adaptive alpha updated to %.3f (dense_frac=%.3f)", ALPHA, dense)
        ALPHA_STATS["dense_clicks"] = 0
        ALPHA_STATS["sparse_clicks"] = 0

# -------------------------
# FastAPI lifespan lock fix and endpoints (optional)
# -------------------------
_LIFESPAN_LOCK = threading.Lock()
if HAS_FASTAPI:
    def _lifespan_start():
        with _LIFESPAN_LOCK:
            ensure_docs_jsonl()
            init_db()
            import_jsonl_to_db(DOCS_JSONL)
            start_background_index()
            if HAS_PROM:
                try:
                    start_http_server(RUNTIME_CONFIG["prometheus_port"])
                    log.info("prometheus metrics available")
                except Exception:
                    pass
            log.info("lifespan startup completed")
    app = FastAPI(title=VERSION, version="7.0-patched-en")
    class BuildResp(BaseModel):
        status: str
        docs_indexed: int
    class SearchReq(BaseModel):
        query: str
        top_k: Optional[int] = 10
        drinks: Optional[List[str]] = []
    class FeedbackReq(BaseModel):
        query: str
        doc_id: str
        clicked: int
    @app.on_event("startup")
    def _startup():
        _lifespan_start()
    @app.get("/status")
    def status():
        return {"index_ready": INDEX_READY.is_set(), "num_docs": len(DOC_TEXTS), "cache": RESULT_CACHE.stats()}
    @app.post("/build_index", response_model=BuildResp)
    def build_index():
        start_background_index()
        return BuildResp(status="started", docs_indexed=len(load_docs_from_db()))
    @app.post("/search")
    def api_search(req: SearchReq, request: Request):
        q = req.query.strip()[:MAX_QUERY_LEN]
        if not q:
            raise HTTPException(status_code=400, detail="empty query")
        client_ip = request.client.host if request.client else None
        try:
            out = search_pipeline(q, top_k=req.top_k or 10, drinks=req.drinks or [], session_state={}, client_ip=client_ip)
            return out
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=429, detail=str(e))
    @app.post("/feedback")
    def feedback(req: FeedbackReq):
        try:
            record_feedback(req.query, req.doc_id, req.clicked)
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("INSERT INTO feedback (ts,query,doc_id,clicked) VALUES (?,?,?,?)", (int(time.time()), req.query, req.doc_id, int(req.clicked)))
            conn.commit()
            conn.close()
            return {"ok": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Demo and CLI
# -------------------------
def run_demo_after_index():
    log.info("demo waiting for index ready...")
    if not INDEX_READY.wait(timeout=60):
        log.warning("index not ready")
        return
    demo_queries = ["tracking","restore","power magnet","invisible","brain activator","doppelganger","tracking plan","energy drink","rollback","locate"]
    out_lines = [f"=== {VERSION} Demo Output ==="]
    for q in demo_queries:
        try:
            res = search_pipeline(q, top_k=5, drinks=["quantum"], session_state={"session_id":"demo"}, client_ip=None)
            out_lines.append(f"Demo query: {q}  (latency_ms={res['meta'].get('latency_ms')})")
            if not res.get("results"):
                out_lines.append("  No confident results (fusion policy may still yield empty).")
            for i, r in enumerate(res.get("results", []), 1):
                out_lines.append(f"  {i}. [{r['id']}] score={r.get('score')} src={r.get('source')} ver={r.get('version')}")
                out_lines.append(f"     {r.get('text')}")
        except Exception as e:
            out_lines.append(f"  Error running demo query '{q}': {e}")
    out_lines.append("=== End Demo ===")
    print("\n".join(out_lines))
    try:
        with open(DEMO_LOG, "a", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
    except Exception:
        pass

def run_benchmarks(sample_queries: List[str] = None, k: int = 5, repeats: int = 3):
    if sample_queries is None:
        sample_queries = ["tracking","restore","power magnet","invisible"]
    results = []
    for q in sample_queries:
        latencies = []
        for _ in range(repeats):
            t0 = time.time()
            out = search_pipeline(q, top_k=k, drinks=["quantum"], session_state={"session_id":"bench"}, client_ip=None)
            latencies.append((time.time() - t0) * 1000.0)
        avg_lat = sum(latencies) / len(latencies)
        results.append({"query": q, "avg_latency_ms": avg_lat, "top_k": k, "results_count": len(out["results"])})
    fname = f"bench_report_v7_patched_en_{int(time.time())}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("benchmark finished, report=%s", fname)
    return results

def find_free_port(start=None, end=None):
    s = start or RUNTIME_CONFIG["port_range_start"]
    e = end or RUNTIME_CONFIG["port_range_end"]
    for p in range(s, e + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port")

def cli_loop(sync_build: bool = False, port: int = None, bench: bool = False):
    ensure_docs_jsonl()
    init_db()
    import_jsonl_to_db(DOCS_JSONL)
    if sync_build:
        start_background_index()
        log.info("waiting for index...")
        INDEX_READY.wait(timeout=120)
    else:
        start_background_index()
    threading.Thread(target=run_demo_after_index, daemon=True).start()
    if bench:
        run_benchmarks()
    print(f"{VERSION} - CLI mode (drinks: {', '.join(RUNTIME_CONFIG['available_drinks'])})")
    session_state = {"session_id": f"sess_{int(time.time())}"}
    try:
        while True:
            try:
                q = input("\nQuery> ").strip()
            except EOFError:
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break
            if "|" in q:
                parts = q.split("|", 1)
                query_text = parts[0].strip()
                drinks = [d.strip() for d in parts[1].split(",") if d.strip()]
            else:
                query_text = q
                drinks = []
            try:
                out = search_pipeline(query_text, top_k=5, drinks=drinks, session_state=session_state)
                meta = out.get("meta", {})
                print(f"alpha={meta.get('alpha')}, latency_ms={meta.get('latency_ms')}, cache_hit={meta.get('cache_hit', False)}")
                if not out.get("results"):
                    print("No confident results after fusion policy.")
                for i, r in enumerate(out.get("results", []), 1):
                    print(f"{i}. [{r['id']}] score={r['score']} tags={r.get('tags')}\n   {r['text']}")
                fb = input("Enter clicked result number (or blank): ").strip()
                if fb.isdigit():
                    idx = int(fb) - 1
                    if 0 <= idx < len(out.get("results", [])):
                        rec = out["results"][idx]
                        record_feedback(query_text, rec["id"], True, session_state)
                        conn = sqlite3.connect(DB_PATH)
                        cur = conn.cursor()
                        cur.execute("INSERT INTO feedback (ts,query,doc_id,clicked) VALUES (?,?,?,?)", (int(time.time()), query_text, rec["id"], 1))
                        conn.commit()
                        conn.close()
            except Exception as e:
                print("Error:", e)
    except KeyboardInterrupt:
        print("\nExiting CLI...")

# -------------------------
# Entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--sync-build", action="store_true", help="wait for index build synchronously on startup")
    parser.add_argument("--port", type=int, default=None, help="port hint")
    parser.add_argument("--bench", action="store_true", help="run benchmark after build")
    args = parser.parse_args()
    ensure_docs_jsonl()
    init_db()
    if not load_docs_from_db():
        import_jsonl_to_db(DOCS_JSONL)
    global CONFIG_HASH
    CONFIG_HASH = compute_config_hash(RUNTIME_CONFIG)
    if HAS_FASTAPI and ("uvicorn" in os.path.basename(sys.argv[0]) or "gunicorn" in os.path.basename(sys.argv[0])):
        log.info("ASGI detected; FastAPI lifespan will manage startup")
        return
    else:
        port = args.port or os.environ.get("PORT")
        if port is None:
            try:
                port = find_free_port(RUNTIME_CONFIG["port_range_start"], RUNTIME_CONFIG["port_range_end"])
            except Exception:
                port = 20000
        cli_loop(sync_build=args.sync_build, port=port, bench=args.bench)

if __name__ == "__main__":
    main()
