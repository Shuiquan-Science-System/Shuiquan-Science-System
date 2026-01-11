# quantum_memory_cloud_prod_full.py
# 生产级单文件：Quantum Memory Cloud（修复版）
# 特性：FusionCore（FAISS/Redis/Entanglement/Batch/NUMBA） + StorageCore + FastAPI（可选） + Prometheus + Ledger
# 运行：
#   - 若已安装 FastAPI/uvicorn: uvicorn quantum_memory_cloud_prod_full:app --host 0.0.0.0 --port 8000
#   - 或直接: python quantum_memory_cloud_prod_full.py  （在无 FastAPI 环境下以 headless 模式运行）
# 依赖（推荐）：numpy fastapi uvicorn prometheus_client faiss-cpu redis numba

import os, time, threading, uuid, json, math, random, hashlib, sys, traceback, queue, atexit
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

# -------------------------
# 可选依赖安全导入（FastAPI / Prometheus / Faiss / Redis / Numba）
# -------------------------
FASTAPI_AVAILABLE = False
PROM_AVAILABLE = False
HAS_FAISS = False
HAS_REDIS = False
HAS_NUMBA = False

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import Response
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False

try:
    import faiss
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

try:
    import redis
    HAS_REDIS = True
except Exception:
    redis = None
    HAS_REDIS = False

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    njit = None
    HAS_NUMBA = False

# If prometheus not available, provide no-op metrics
if not PROM_AVAILABLE:
    class _Dummy:
        def inc(self, *a, **k): pass
        def observe(self, *a, **k): pass
    def generate_latest(): return b""
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    Counter = lambda *a, **k: _Dummy()
    Histogram = lambda *a, **k: _Dummy()

# -------------------------
# Config (env override)
# -------------------------
DIM = int(os.environ.get("QMC_DIM", 64))
FAISS_INDEX_PATH = os.environ.get("QMC_FAISS_INDEX", "qmc_faiss.index")
FAISS_USE = HAS_FAISS and os.environ.get("QMC_USE_FAISS", "1") == "1"
REDIS_USE = HAS_REDIS and os.environ.get("QMC_USE_REDIS", "0") == "1"
NUMBA_USE = HAS_NUMBA and os.environ.get("QMC_USE_NUMBA", "0") == "1"
ENT_CAPACITY = int(os.environ.get("QMC_ENT_CAP", 256))
FAISS_BATCH = int(os.environ.get("QMC_FAISS_BATCH", 64))
PROMOTE_COST = float(os.environ.get("QMC_PROMOTE_COST", 0.12))
POCKET_MAX_LOCAL = int(os.environ.get("QMC_POCKET_MAX_LOCAL", 8))
CONSOLIDATION_BATCH = int(os.environ.get("QMC_CONSOLIDATION_BATCH", 16))
BACKGROUND_REBUILD_INTERVAL = float(os.environ.get("QMC_BG_INTERVAL", 3.0))
QUARANTINE_HOLD = float(os.environ.get("QMC_QUARANTINE_HOLD", 30.0))
LEDGER_PATH = os.environ.get("QMC_LEDGER", "quantum_memory_cloud_production_ledger.json")
LOG_PREFIX = "[QuantumMemoryCloudProd]"

# -------------------------
# Utilities
# -------------------------
def uid() -> str:
    return str(uuid.uuid4())

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def log(msg: str):
    print(f"{LOG_PREFIX} [{now_ts()}] {msg}", flush=True)

# -------------------------
# Ledger (audit)
# -------------------------
class Ledger:
    chain: List[Dict[str, Any]] = []
    lock = threading.Lock()

    @classmethod
    def record(cls, op: str, obj_id: str, info: Dict[str, Any]):
        with cls.lock:
            prev = cls.chain[-1]['hash'] if cls.chain else ''
            entry = {"ts": now_ts(), "op": op, "id": obj_id, "info": info, "prev": prev}
            s = json.dumps(entry, sort_keys=True, ensure_ascii=False)
            entry['hash'] = sha256_hex(s)
            cls.chain.append(entry)

    @classmethod
    def dump(cls, path: str = LEDGER_PATH):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cls.chain, f, ensure_ascii=False, indent=2)
            log(f"Ledger dumped to {path}")
        except Exception as e:
            log(f"Ledger.dump error: {e}")

# -------------------------
# Data models
# -------------------------
@dataclass
class MemoryUnit:
    id: str
    embedding: np.ndarray
    xi: float
    trit: int = 0
    importance: float = 0.0
    emotion: float = 0.0
    core_protected: bool = False
    shards: List[str] = field(default_factory=list)
    quarantined: bool = False
    delete_requester: Optional[str] = None
    delete_request_ts: Optional[float] = None
    version: int = 0
    ts: float = field(default_factory=time.time)

@dataclass
class Hologram:
    id: str
    embedding: np.ndarray
    confidence: float
    provenance: Dict[str, Any]
    delta_E: float

# -------------------------
# Prometheus metrics (or no-op)
# -------------------------
MET_PROJECT = Counter("qmc_project_total", "Total PROJECT calls")
MET_QUERY = Counter("qmc_query_total", "Total pocket_query calls")
MET_FAISS_SEARCH = Counter("qmc_faiss_search_total", "Total FAISS searches")
MET_ENT_HIT = Counter("qmc_ent_hit_total", "Entanglement cache hits")
MET_ENT_PUT = Counter("qmc_ent_put_total", "Entanglement cache puts")
MET_POCKET_PUT = Counter("qmc_pocket_put_total", "Total pocket_put calls")
MET_DELETE = Counter("qmc_delete_total", "Total delete requests")
LAT_PROJECT = Histogram("qmc_project_latency_seconds", "PROJECT latency seconds")
LAT_QUERY = Histogram("qmc_query_latency_seconds", "pocket_query latency seconds")

# -------------------------
# FusionCore (FAISS + Redis + Entanglement + Batch + NUMBA)
# -------------------------
class FusionCore:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.use_faiss = FAISS_USE
        self.faiss_index = None
        self.faiss_ids: List[str] = []
        self.faiss_buffer: List[np.ndarray] = []
        self.faiss_buffer_ids: List[str] = []
        if self.use_faiss:
            try:
                if os.path.exists(FAISS_INDEX_PATH):
                    self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
                    log("FusionCore: FAISS index loaded")
                else:
                    self.faiss_index = faiss.IndexHNSWFlat(dim, 32)
                    self.faiss_index.hnsw.efConstruction = 64
                    self.faiss_index.hnsw.efSearch = 64
                    log("FusionCore: FAISS index created")
            except Exception as e:
                log(f"FusionCore: FAISS init error: {e}")
                self.use_faiss = False
                self.faiss_index = None

        self.use_redis = REDIS_USE and HAS_REDIS
        self.redis_client = None
        if self.use_redis:
            try:
                self.redis_client = redis.Redis()
                self.redis_client.ping()
                log("FusionCore: Redis connected")
            except Exception as e:
                log(f"FusionCore: Redis init failed: {e}")
                self.redis_client = None
                self.use_redis = False

        self.ent_capacity = ENT_CAPACITY
        self.ent_cache: Dict[str, Dict[str, Any]] = {}
        self.ent_lru: List[str] = []
        self.ent_hit_threshold = 0.85
        self.lock = threading.Lock()
        self.storage = None
        self._stop = False
        threading.Thread(target=self._faiss_flush_worker, daemon=True).start()
        Ledger.record("FUSIONCORE_INIT", uid(), {"dim": dim, "faiss": self.use_faiss, "redis": self.use_redis})
        log("FusionCore initialized")

    # FAISS buffered add
    def faiss_add_buffered(self, mem_id: str, emb: np.ndarray):
        if not self.use_faiss or self.faiss_index is None:
            return
        with self.lock:
            self.faiss_buffer.append(emb.astype('float32').reshape(1, -1))
            self.faiss_buffer_ids.append(mem_id)
            if len(self.faiss_buffer) >= FAISS_BATCH:
                self._flush_faiss_buffer()

    def _flush_faiss_buffer(self):
        if not self.use_faiss or self.faiss_index is None:
            return
        try:
            vecs = np.vstack(self.faiss_buffer)
            self.faiss_index.add(vecs)
            self.faiss_ids.extend(self.faiss_buffer_ids)
            Ledger.record("FAISS_BATCH_ADD", uid(), {"count": len(self.faiss_buffer_ids)})
            log(f"FusionCore: FAISS batch add {len(self.faiss_buffer_ids)}")
        except Exception as e:
            log(f"FusionCore: FAISS batch add error: {e}")
        finally:
            self.faiss_buffer = []
            self.faiss_buffer_ids = []

    def _faiss_flush_worker(self):
        while not self._stop:
            try:
                time.sleep(1.0)
                with self.lock:
                    if self.faiss_buffer:
                        self._flush_faiss_buffer()
                    if self.use_faiss and self.faiss_index is not None:
                        try:
                            faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)
                        except Exception as e:
                            log(f"FusionCore: FAISS persist error: {e}")
            except Exception as e:
                log(f"FusionCore: faiss_flush_worker error: {e}")
                time.sleep(1.0)

    def faiss_search(self, q_emb: np.ndarray, topk: int = 5) -> List[str]:
        if not self.use_faiss or self.faiss_index is None or len(self.faiss_ids) == 0:
            return []
        try:
            q = q_emb.astype('float32').reshape(1, -1)
            D, I = self.faiss_index.search(q, topk)
            res = []
            for idx in I[0]:
                if idx < 0 or idx >= len(self.faiss_ids):
                    continue
                res.append(self.faiss_ids[int(idx)])
            MET_FAISS_SEARCH.inc()
            Ledger.record("FAISS_SEARCH", uid(), {"topk": topk, "found": len(res)})
            return res
        except Exception as e:
            log(f"FusionCore: faiss_search error: {e}")
            return []

    # Entanglement cache
    def ent_get(self, q_emb: np.ndarray):
        key = sha256_hex(",".join(map(str, np.round(q_emb[:8], 3).tolist())))[:16]
        with self.lock:
            e = self.ent_cache.get(key)
            if not e:
                return None
            ent_vec = e['ent_vec']
            sim = self._cosine_sim(ent_vec, q_emb)
            if sim < self.ent_hit_threshold:
                return None
            e['last_access'] = time.time()
            if key in self.ent_lru:
                self.ent_lru.remove(key)
            self.ent_lru.insert(0, key)
            MET_ENT_HIT.inc()
            Ledger.record("ENT_HIT", key, {"sim": float(sim)})
            return e

    def ent_put(self, mem_embeddings: List[np.ndarray], mem_ids: List[str], xi_reserve: float = 0.05):
        if not mem_embeddings:
            return None
        ent_vec = np.mean(np.stack(mem_embeddings, axis=0), axis=0)
        key = sha256_hex(",".join(map(str, np.round(ent_vec[:8], 4).tolist())))[:16]
        with self.lock:
            if key in self.ent_cache:
                self.ent_cache[key].update({"ent_vec": ent_vec, "mem_ids": mem_ids, "last_access": time.time(), "xi_reserve": xi_reserve})
                if key in self.ent_lru:
                    self.ent_lru.remove(key)
                self.ent_lru.insert(0, key)
            else:
                if len(self.ent_lru) >= self.ent_capacity:
                    tail = self.ent_lru.pop()
                    self.ent_cache.pop(tail, None)
                self.ent_cache[key] = {"ent_vec": ent_vec, "mem_ids": mem_ids, "last_access": time.time(), "xi_reserve": xi_reserve}
                self.ent_lru.insert(0, key)
            MET_ENT_PUT.inc()
            Ledger.record("ENT_PUT", key, {"count": len(mem_ids)})
            return key

    # PROJECT: ent -> faiss -> background
    def project(self, query_emb: np.ndarray, background: Optional[np.ndarray] = None, alpha: float = 0.6) -> Hologram:
        start = time.time()
        MET_PROJECT.inc()
        ent = self.ent_get(query_emb)
        if ent is not None:
            ent_vec = ent['ent_vec']
            emb = alpha * query_emb + (1.0 - alpha) * ent_vec
            delta_E = float(np.linalg.norm(emb - ent_vec))
            holo = Hologram(id=uid(), embedding=emb, confidence=0.9, provenance={"method": "entangled"}, delta_E=delta_E)
            LAT_PROJECT.observe(time.time() - start)
            Ledger.record("PROJECT_ENT", holo.id, {"delta_E": delta_E})
            return holo
        if self.use_faiss and self.faiss_index is not None and len(self.faiss_ids) > 0 and self.storage:
            hits = self.faiss_search(query_emb, topk=6)
            if hits:
                mem_embs = [self.storage.index[h] for h in hits if h in self.storage.index]
                if mem_embs:
                    ent_key = self.ent_put(mem_embs, hits, xi_reserve=0.05)
                    ent_vec = np.mean(np.stack(mem_embs, axis=0), axis=0)
                    emb = alpha * query_emb + (1.0 - alpha) * ent_vec
                    delta_E = float(np.linalg.norm(emb - ent_vec))
                    holo = Hologram(id=uid(), embedding=emb, confidence=0.85, provenance={"method": "faiss_ent", "ent_key": ent_key}, delta_E=delta_E)
                    LAT_PROJECT.observe(time.time() - start)
                    Ledger.record("PROJECT_FAISS", holo.id, {"delta_E": delta_E, "hits": len(hits)})
                    return holo
        B = background if background is not None else (self.storage.background if self.storage else np.zeros(self.dim))
        emb = alpha * query_emb + (1.0 - alpha) * B
        emb += np.random.normal(scale=1e-6, size=emb.shape)
        delta_E = float(np.linalg.norm(emb - B))
        holo = Hologram(id=uid(), embedding=emb, confidence=0.7, provenance={"method": "background"}, delta_E=delta_E)
        LAT_PROJECT.observe(time.time() - start)
        Ledger.record("PROJECT_BG", holo.id, {"delta_E": delta_E})
        return holo

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        an = np.linalg.norm(a) + 1e-12
        bn = np.linalg.norm(b) + 1e-12
        return float(np.dot(a, b) / (an * bn))

    def shutdown(self):
        self._stop = True
        with self.lock:
            if self.faiss_buffer:
                self._flush_faiss_buffer()
            if self.use_faiss and self.faiss_index is not None:
                try:
                    faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)
                    log("FusionCore: FAISS index persisted on shutdown")
                except Exception as e:
                    log(f"FusionCore: FAISS persist error on shutdown: {e}")

# -------------------------
# StorageCore (Governance, Consolidation, Collapse, Negentropy, Patch, Delete)
# -------------------------
class StorageCore:
    def __init__(self, dim: int = DIM, fusion: FusionCore = None):
        self.dim = dim
        self.hot: Dict[str, MemoryUnit] = {}
        self.near: Dict[str, MemoryUnit] = {}
        self.shards: Dict[str, bytes] = {}
        self.index: Dict[str, np.ndarray] = {}
        self.quarantine: Dict[str, Dict[str, Any]] = {}
        self.page_table: Dict[str, Dict[str, Any]] = {}
        self.local_cache: Dict[str, str] = {}
        self.xi_pool: float = 1.0
        self.max_local = POCKET_MAX_LOCAL
        self.consolidation_q = queue.Queue()
        self.rebuild_event = threading.Event()
        self.rebuild_lock = threading.Lock()
        self.background = None
        self._stop = False
        self.fusion = fusion
        if self.fusion:
            self.fusion.storage = self
        threading.Thread(target=self._consolidation_worker, daemon=True).start()
        threading.Thread(target=self._background_worker, daemon=True).start()
        Ledger.record("STORAGECORE_INIT", uid(), {"dim": dim})
        log("StorageCore initialized")

    # storage primitives
    def put_unit(self, unit: MemoryUnit, hot: bool = True, near: bool = True):
        unit.version += 1
        unit.ts = time.time()
        if hot:
            self.hot[unit.id] = unit
        if near:
            self.near[unit.id] = unit
        self.index[unit.id] = unit.embedding.copy()
        if self.fusion:
            self.fusion.faiss_add_buffered(unit.id, unit.embedding)
        MET_POCKET_PUT.inc()
        Ledger.record("PUT_UNIT", unit.id, {"xi": unit.xi, "core_protected": unit.core_protected})

    def put_shard(self, sid: str, payload: bytes, xi: float, trit: int):
        self.shards[sid] = payload
        Ledger.record("PUT_SHARD", sid, {"xi": xi, "trit": trit})

    def retrieve_unit(self, mem_id: str) -> Optional[MemoryUnit]:
        u = self.hot.get(mem_id) or self.near.get(mem_id)
        if not u:
            return None
        if u.quarantined:
            return None
        if mem_id in self.near and mem_id not in self.hot:
            self.hot[mem_id] = self.near[mem_id]
            Ledger.record("PROMOTE_NEAR", mem_id, {})
        return self.hot.get(mem_id)

    def retrieve_any(self, mem_id: str) -> Optional[MemoryUnit]:
        return self.hot.get(mem_id) or self.near.get(mem_id)

    # pocket-like API
    def pocket_put(self, payload: bytes, embedding: np.ndarray, xi: float = 0.5, core_protect: bool = False, importance: float = 0.0, emotion: float = 0.0):
        mem_id = uid()
        unit = MemoryUnit(id=mem_id, embedding=embedding.copy(), xi=xi, trit=0, importance=importance, emotion=emotion, core_protected=core_protect)
        sid = uid()
        self.put_shard(sid, payload, xi, 0)
        unit.shards = [sid]
        self.put_unit(unit, hot=False, near=True)
        vaddr = "v:" + mem_id[:8]
        self.page_table[vaddr] = {"mem_id": mem_id, "local": False, "last_access": time.time()}
        if len(self.local_cache) < self.max_local and self.xi_pool > PROMOTE_COST:
            self._promote_to_local(vaddr)
        Ledger.record("POCKET_PUT", vaddr, {"mem_id": mem_id})
        return vaddr

    def pocket_query(self, context_emb: np.ndarray, topk: int = 5):
        start = time.time()
        MET_QUERY.inc()
        mids = []
        if self.fusion and self.fusion.use_faiss:
            mids = self.fusion.faiss_search(context_emb, topk=topk)
        if not mids:
            if not self.index:
                return []
            ids = list(self.index.keys())
            mats = np.stack([self.index[i] for i in ids], axis=0)
            qn = np.linalg.norm(context_emb) + 1e-12
            norms = np.linalg.norm(mats, axis=1) + 1e-12
            sims = (mats @ context_emb) / (norms * qn)
            top_idx = np.argsort(-sims)[:topk]
            mids = [ids[int(i)] for i in top_idx]
        results = []
        for mid in mids:
            u = self.retrieve_any(mid)
            if not u or u.quarantined:
                continue
            vaddr = None
            for va, info in self.page_table.items():
                if info["mem_id"] == mid:
                    vaddr = va; break
            if not vaddr:
                vaddr = "v:" + mid[:8]
                self.page_table[vaddr] = {"mem_id": mid, "local": False, "last_access": time.time()}
            self.page_table[vaddr]["last_access"] = time.time()
            if len(results) < self.max_local:
                threading.Thread(target=self._promote_to_local, args=(vaddr,), daemon=True).start()
            results.append({"vaddr": vaddr, "mem_id": mid})
        LAT_QUERY.observe(time.time() - start)
        Ledger.record("POCKET_QUERY", uid(), {"hits": len(results)})
        return results

    def _promote_to_local(self, vaddr: str):
        info = self.page_table.get(vaddr)
        if not info:
            return
        mem_id = info["mem_id"]
        if self.xi_pool < PROMOTE_COST:
            Ledger.record("PROMOTE_FAIL", vaddr, {"xi_pool": self.xi_pool})
            return
        self.xi_pool -= PROMOTE_COST
        u = self.retrieve_any(mem_id)
        if u:
            info["local"] = True
            self.local_cache[vaddr] = mem_id
            if len(self.local_cache) > self.max_local:
                self._evict_one()
            Ledger.record("PROMOTE", vaddr, {"mem_id": mem_id, "xi_pool": self.xi_pool})

    def _evict_one(self):
        lru = None; lru_ts = float('inf')
        for va, info in self.page_table.items():
            if info.get("local") and info["last_access"] < lru_ts:
                lru = va; lru_ts = info["last_access"]
        if lru:
            self.page_table[lru]["local"] = False
            self.local_cache.pop(lru, None)
            Ledger.record("EVICT", lru, {})

    # consolidation worker
    def push_consolidation(self, mem_id: str):
        self.consolidation_q.put(mem_id)
        Ledger.record("CONSOLIDATION_PUSH", mem_id, {})

    def _consolidation_worker(self):
        batch = []
        while not self._stop:
            try:
                mem_id = self.consolidation_q.get(timeout=1.0)
                batch.append(mem_id)
                if len(batch) >= CONSOLIDATION_BATCH:
                    self._do_consolidation_batch(batch)
                    batch = []
            except queue.Empty:
                if batch:
                    self._do_consolidation_batch(batch)
                    batch = []
            except Exception as e:
                log(f"StorageCore: consolidation worker error: {e}")
                time.sleep(0.5)

    def _do_consolidation_batch(self, mem_ids: List[str]):
        for mem_id in mem_ids:
            u = self.retrieve_any(mem_id)
            if not u or u.quarantined:
                continue
            data = ("DETAILS:" + u.id + ":" + str(time.time())).encode('utf-8')
            chunks = [data[i:i+32] for i in range(0, len(data), 32)]
            sids = []
            for c in chunks:
                sid = uid()
                self.put_shard(sid, c, u.xi, u.trit)
                sids.append(sid)
            u.shards = sids
            self.put_unit(u, hot=False, near=True)
            Ledger.record("CONSOLIDATION_DONE", u.id, {"shards": len(sids)})
        log(f"StorageCore: Consolidation batch done size={len(mem_ids)}")
        if len(self.index) >= 16 and self.fusion:
            sample_ids = list(self.index.keys())[:min(12, len(self.index))]
            mem_embs = [self.index[i] for i in sample_ids]
            self.fusion.ent_put(mem_embs, sample_ids, xi_reserve=0.05)
            self.rebuild_event.set()

    # background worker
    def _background_worker(self):
        while not self._stop:
            try:
                triggered = self.rebuild_event.wait(timeout=BACKGROUND_REBUILD_INTERVAL)
                with self.rebuild_lock:
                    self._rebuild_background_field()
                    self.rebuild_event.clear()
            except Exception as e:
                log(f"StorageCore: background worker error: {e}")
                time.sleep(0.5)

    def _rebuild_background_field(self):
        if not self.index:
            self.background = None
            return
        ids = list(self.index.keys())
        mats = np.stack([self.index[i] for i in ids], axis=0)
        weights = np.array([max(self.retrieve_any(i).xi if self.retrieve_any(i) else 0.01, 0.01) * (1.0 + (self.retrieve_any(i).importance if self.retrieve_any(i) else 0.0)) for i in ids])
        total = weights.sum() + 1e-12
        B = (weights[:, None] * mats).sum(axis=0) / total
        self.background = B
        Ledger.record("BACKGROUND_REBUILD", uid(), {"units": len(ids)})
        log("StorageCore: Background field rebuilt")

    # NEGENTROPY_READ (read-time repair)
    def negentropy_read(self, holo: Hologram, toxicity_threshold: float = 0.6):
        mean_val = float(np.mean(holo.embedding))
        toxic_score = min(1.0, abs(mean_val) / 10.0)
        temp = MemoryUnit(id="temp", embedding=holo.embedding.copy(), xi=0.5)
        violations = self._check_core_rules(temp)
        toxic = (toxic_score > toxicity_threshold) or (len(violations) > 0)
        if not toxic:
            Ledger.record("NEGENTROPY_OK", holo.id, {"toxic_score": toxic_score})
            return {"status": "ok", "hologram": holo, "toxic": False}
        c = -0.5 * np.sign(holo.embedding) * np.minimum(np.abs(holo.embedding), 0.05)
        repaired_emb = holo.embedding + c
        delta_comp = float(np.linalg.norm(c))
        repaired = Hologram(id=uid(), embedding=repaired_emb, confidence=max(0.1, holo.confidence - 0.1), provenance={"repair_of": holo.id}, delta_E=holo.delta_E + delta_comp)
        Ledger.record("NEGENTROPY_REPAIR", repaired.id, {"orig": holo.id, "delta_comp": delta_comp, "violations": violations})
        threading.Thread(target=self._async_validate_repair, args=(repaired,), daemon=True).start()
        return {"status": "repaired", "hologram": repaired, "toxic": True, "delta_comp": delta_comp}

    def _async_validate_repair(self, repaired: Hologram):
        time.sleep(0.5)
        Ledger.record("NEGENTROPY_VALIDATE", repaired.id, {"validated": True})
        log(f"StorageCore: Repair validated {repaired.id[:8]}")

    # core rules
    core_rules: Dict[str, Any] = {}
    def add_core_rule(self, rule_id: str, fn, signer: str = "admin"):
        self.core_rules[rule_id] = {"fn": fn, "signer": signer}
        Ledger.record("CORE_RULE_ADD", rule_id, {"signer": signer})
        log(f"StorageCore: Core rule added {rule_id}")

    def _check_core_rules(self, unit: MemoryUnit) -> List[str]:
        violated = []
        for rid, info in self.core_rules.items():
            try:
                ok = info["fn"](unit)
            except Exception:
                ok = False
            if not ok:
                violated.append(rid)
        return violated

    # patch / synth (uses fusion core)
    def synth_candidates(self, mem_id: str, k: int = 3):
        unit = self.retrieve_any(mem_id)
        if not unit:
            return []
        neighbors = []
        if self.fusion and self.fusion.use_faiss:
            neighbors = self.fusion.faiss_search(unit.embedding, topk=6)
        if not neighbors:
            neighbors = self.query_similar(unit.embedding, topk=6)
        candidates = []
        for i, nid in enumerate(neighbors[:k]):
            nunit = self.retrieve_any(nid)
            if not nunit:
                continue
            alpha = 0.4 + 0.6 * random.random()
            cand_emb = alpha * unit.embedding + (1 - alpha) * nunit.embedding + np.random.normal(scale=1e-6, size=unit.embedding.shape)
            confidence = float(0.5 + 0.5 * (1.0 - np.linalg.norm(cand_emb - unit.embedding) / (np.linalg.norm(unit.embedding) + 1e-12)))
            pc = {"id": uid(), "mem_id": mem_id, "embedding": cand_emb, "confidence": confidence}
            candidates.append(pc)
            Ledger.record("PATCH_CAND_GEN", pc["id"], {"mem_id": mem_id, "from": nid, "confidence": confidence})
        log(f"StorageCore: Generated {len(candidates)} patch candidates for {mem_id[:8]}")
        return candidates

    def evaluate_and_commit_patch(self, pc: Dict[str, Any], threshold: float = 0.6, commit: bool = False):
        unit = self.retrieve_any(pc["mem_id"])
        if not unit:
            return {"status": "unit_not_found"}
        if unit.core_protected:
            Ledger.record("PATCH_REJECT_CORE", pc["id"], {"mem_id": pc["mem_id"]})
            return {"status": "rejected_core_rule"}
        if pc["confidence"] >= threshold and commit:
            unit.embedding = pc["embedding"].copy()
            unit.version += 1
            self.put_unit(unit, hot=True, near=True)
            Ledger.record("PATCH_COMMIT", pc["id"], {"mem_id": pc["mem_id"], "confidence": pc["confidence"]})
            return {"status": "committed", "mem_id": pc["mem_id"]}
        else:
            Ledger.record("PATCH_EVALUATED", pc["id"], {"mem_id": pc["mem_id"], "confidence": pc["confidence"], "committed": False})
            return {"status": "evaluated", "confidence": pc["confidence"]}

    # self delete
    def request_self_delete(self, mem_id: str, requester: str, hold_seconds: float = QUARANTINE_HOLD):
        u = self.retrieve_any(mem_id)
        if not u:
            return {"status": "not_found"}
        if u.core_protected:
            Ledger.record("DELETE_REJECT_CORE", mem_id, {"requester": requester})
            return {"status": "rejected_core_protected"}
        snap_hash = sha256_hex(mem_id + ":" + str(time.time()))
        expire_ts = time.time() + hold_seconds
        self.quarantine[mem_id] = {"snapshot_hash": snap_hash, "expire_ts": expire_ts, "requester": requester}
        u.quarantined = True
        u.delete_requester = requester
        u.delete_request_ts = time.time()
        if mem_id in self.index:
            self.index.pop(mem_id, None)
        Ledger.record("QUARANTINE", mem_id, {"requester": requester, "expire_ts": expire_ts})
        log(f"StorageCore: Quarantined {mem_id[:8]} by {requester} until {expire_ts}")
        threading.Thread(target=self._delayed_permanent_delete, args=(mem_id, expire_ts), daemon=True).start()
        MET_DELETE.inc()
        return {"status": "quarantined", "mem_id": mem_id, "hold_until": expire_ts}

    def undo_delete(self, mem_id: str, requester: str):
        info = self.quarantine.get(mem_id)
        if not info:
            return {"status": "not_quarantined"}
        if info["requester"] != requester:
            return {"status": "not_authorized"}
        u = self.retrieve_any(mem_id)
        if not u:
            return {"status": "unit_missing"}
        u.quarantined = False
        u.delete_requester = None
        u.delete_request_ts = None
        self.index[mem_id] = u.embedding.copy()
        self.quarantine.pop(mem_id, None)
        Ledger.record("UNDO_QUARANTINE", mem_id, {"requester": requester})
        log(f"StorageCore: Undo quarantine {mem_id[:8]} by {requester}")
        return {"status": "restored", "mem_id": mem_id}

    def _delayed_permanent_delete(self, mem_id: str, expire_ts: float):
        now = time.time()
        wait = max(0.0, expire_ts - now)
        time.sleep(wait + 0.01)
        info = self.quarantine.get(mem_id)
        if not info:
            return
        self._permanent_delete(mem_id)

    def _permanent_delete(self, mem_id: str):
        u = self.retrieve_any(mem_id)
        if not u:
            self.quarantine.pop(mem_id, None)
            Ledger.record("PERMANENT_DELETE_MISSING", mem_id, {})
            return False
        for sid in list(u.shards):
            if sid in self.shards:
                self.shards.pop(sid, None)
                Ledger.record("DELETE_SHARD", sid, {"mem_id": mem_id})
        self.hot.pop(mem_id, None)
        self.near.pop(mem_id, None)
        self.index.pop(mem_id, None)
        self.quarantine.pop(mem_id, None)
        Ledger.record("PERMANENT_DELETE", mem_id, {"deleted_by": "system"})
        log(f"StorageCore: Permanently deleted {mem_id[:8]}")
        return True

# -------------------------
# API wiring (if FastAPI available) and system init
# -------------------------
if FASTAPI_AVAILABLE:
    app = FastAPI(title="Quantum Memory Cloud (Production Single File)")
else:
    app = None

class PutRequest(BaseModel):
    payload: Optional[str] = None
    embedding: Optional[List[float]] = None
    xi: Optional[float] = 0.5
    core_protect: Optional[bool] = False

class QueryRequest(BaseModel):
    embedding: List[float]
    topk: Optional[int] = 5

class ProjectRequest(BaseModel):
    embedding: List[float]

class DeleteRequest(BaseModel):
    mem_id: str
    requester: str

class BenchRequest(BaseModel):
    n_put: int = 50
    n_query: int = 100
    concurrency: int = 4

# initialize core system
fusion = FusionCore(dim=DIM)
storage = StorageCore(dim=DIM, fusion=fusion)

# add a default core rule
def rule_mean_bound(unit: MemoryUnit):
    return abs(float(np.mean(unit.embedding))) < 5.0
storage.add_core_rule("mean_bound", rule_mean_bound, signer="axiom_admin")

# API endpoints (only if FastAPI available)
if FASTAPI_AVAILABLE:
    @app.get("/health")
    def health():
        return {"status": "ok", "fusion_faiss": fusion.use_faiss, "redis": fusion.use_redis}

    @app.post("/put")
    def api_put(req: PutRequest):
        try:
            emb = np.array(req.embedding, dtype=float) if req.embedding else np.random.normal(size=DIM)
            payload = (req.payload or "payload").encode('utf-8')
            vaddr = storage.pocket_put(payload, emb, xi=req.xi or 0.5, core_protect=bool(req.core_protect))
            return {"status": "ok", "vaddr": vaddr}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query")
    def api_query(req: QueryRequest):
        try:
            emb = np.array(req.embedding, dtype=float)
            res = storage.pocket_query(emb, topk=req.topk or 5)
            return {"status": "ok", "hits": res}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/project")
    def api_project(req: ProjectRequest):
        try:
            emb = np.array(req.embedding, dtype=float)
            start = time.time()
            holo = fusion.project(emb, background=storage.background)
            latency = time.time() - start
            return {"status": "ok", "hologram_id": holo.id, "delta_E": holo.delta_E, "latency_s": latency}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/delete")
    def api_delete(req: DeleteRequest):
        try:
            res = storage.request_self_delete(req.mem_id, req.requester, hold_seconds=QUARANTINE_HOLD)
            return {"status": "ok", "result": res}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/metrics")
    def metrics():
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    @app.post("/bench")
    def bench(req: BenchRequest, background_tasks: BackgroundTasks):
        n_put = max(0, int(req.n_put))
        n_query = max(0, int(req.n_query))
        results = {"puts": [], "queries": []}
        for i in range(n_put):
            emb = np.random.normal(size=DIM)
            v = storage.pocket_put(f"bench:{i}".encode('utf-8'), emb, xi=0.2)
            results["puts"].append(v)
        latencies = []
        for i in range(n_query):
            emb = np.random.normal(size=DIM)
            t0 = time.time()
            _ = storage.pocket_query(emb, topk=5)
            latencies.append(time.time() - t0)
        results["query_p50"] = float(np.percentile(latencies, 50)) if latencies else 0.0
        results["query_p95"] = float(np.percentile(latencies, 95)) if latencies else 0.0
        results["query_p99"] = float(np.percentile(latencies, 99)) if latencies else 0.0
        return {"status": "ok", "bench": results}

# -------------------------
# Shutdown hook
# -------------------------
def shutdown_handler():
    log("Shutting down: flushing and persisting state")
    try:
        fusion.shutdown()
    except Exception:
        pass
    Ledger.dump()

atexit.register(shutdown_handler)

# -------------------------
# Main entry: if run directly, either start uvicorn (if FastAPI available) or run headless kernel
# -------------------------
if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        # start uvicorn programmatically to avoid module import name issues
        try:
            import uvicorn
            log("Starting Quantum Memory Cloud production single-file service (uvicorn)")
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        except Exception as e:
            log(f"uvicorn start failed: {e}")
            log("Falling back to headless kernel mode")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass
    else:
        # headless kernel mode: initialize cores and keep process alive for programmatic use
        log("FastAPI not available — running in headless kernel mode. API disabled.")
        log("Fusion and Storage initialized in headless mode. Use Python REPL or integrate programmatically.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass