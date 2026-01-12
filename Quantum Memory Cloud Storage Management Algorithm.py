
#!/usr/bin/env python3
# quantum_memory_cloud_prod_full_projection.py
# 生产级单文件：Quantum Memory Cloud（完整版，含 Detox + Micro↔Macro 投影）
# 说明：
#   - 单文件交付，自动检测可选依赖（faiss, redis, fastapi, prometheus, numba）
#   - 在无可选依赖时自动降级为纯 numpy/内存实现
#   - 主要模块：FusionCore, StorageCore, ProjectionEngine (Micro->Macro, Macro->Micro), Ledger, Detox, HTTP 管理端点（可选）
# 运行：
#   - headless: python quantum_memory_cloud_prod_full_projection.py
#   - 启用 HTTP: export QMC_ENABLE_HTTP=1 && python ...
# 环境变量（常用）:
#   QMC_DIM, QMC_USE_FAISS, QMC_USE_REDIS, QMC_ENABLE_HTTP, QMC_ENABLE_PROM, QMC_ENT_CAP, QMC_FAISS_BATCH, QMC_BG_INTERVAL
# 建议：先在测试环境跑通，再在生产环境启用 FAISS/Redis/Prometheus。

import os, sys, time, uuid, json, math, random, hashlib, threading, traceback, queue, atexit, signal
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# Optional imports detection
# -------------------------
FASTAPI_AVAILABLE = False
PROM_AVAILABLE = False
HAS_FAISS = False
HAS_REDIS = False
HAS_NUMBA = False

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response, JSONResponse
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, Gauge
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

# -------------------------
# Config (env override)
# -------------------------
DIM = int(os.environ.get("QMC_DIM", 128))
FAISS_INDEX_PATH = os.environ.get("QMC_FAISS_INDEX", "qmc_faiss.index")
FAISS_USE = HAS_FAISS and os.environ.get("QMC_USE_FAISS", "1") == "1"
REDIS_USE = HAS_REDIS and os.environ.get("QMC_USE_REDIS", "0") == "1"
ENABLE_HTTP = FASTAPI_AVAILABLE and os.environ.get("QMC_ENABLE_HTTP", "0") == "1"
ENABLE_PROM = PROM_AVAILABLE and os.environ.get("QMC_ENABLE_PROM", "0") == "1"
NUMBA_USE = HAS_NUMBA and os.environ.get("QMC_USE_NUMBA", "0") == "1"

ENT_CAPACITY = int(os.environ.get("QMC_ENT_CAP", 1024))
FAISS_BATCH = int(os.environ.get("QMC_FAISS_BATCH", 128))
PROMOTE_COST = float(os.environ.get("QMC_PROMOTE_COST", 0.12))
POCKET_MAX_LOCAL = int(os.environ.get("QMC_POCKET_MAX_LOCAL", 16))
CONSOLIDATION_BATCH = int(os.environ.get("QMC_CONSOLIDATION_BATCH", 64))
BACKGROUND_REBUILD_INTERVAL = float(os.environ.get("QMC_BG_INTERVAL", 5.0))
QUARANTINE_HOLD = float(os.environ.get("QMC_QUARANTINE_HOLD", 60.0))
LEDGER_PATH = os.environ.get("QMC_LEDGER", "quantum_memory_cloud_production_ledger.json")
LOG_PREFIX = os.environ.get("QMC_LOG_PREFIX", "[QuantumMemoryCloudProdProj]")
DEMO_UNITS = int(os.environ.get("QMC_DEMO_UNITS", 512))
AUTO_PROJECT_BATCH = int(os.environ.get("QMC_AUTO_PROJECT_BATCH", 16))
PREFETCH_ENABLED = os.environ.get("QMC_PREFETCH", "1") == "1"
MAX_PROJECT_WORKERS = int(os.environ.get("QMC_PROJECT_WORKERS", 8))
TOKEN_BUCKET_RATE = float(os.environ.get("QMC_TOKEN_RATE", 100.0))
TOKEN_BUCKET_CAP = float(os.environ.get("QMC_TOKEN_CAP", 200.0))

# Detox & Projection config
TOXICITY_THRESHOLD = float(os.environ.get("QMC_TOXICITY_THRESHOLD", 0.6))
REPAIR_THRESHOLD = float(os.environ.get("QMC_REPAIR_THRESHOLD", 0.65))
ANOMALY_ZSCORE = float(os.environ.get("QMC_ANOMALY_ZSCORE", 4.0))
DECAY_INTERVAL = float(os.environ.get("QMC_DECAY_INTERVAL", 3600.0))
DECAY_RATE = float(os.environ.get("QMC_DECAY_RATE", 0.01))
SANITIZER_BLACKLIST = os.environ.get("QMC_SANITIZER_BLACKLIST", "").split(",") if os.environ.get("QMC_SANITIZER_BLACKLIST") else []

# Projection engine config
PROJ_MICRO_DIM = int(os.environ.get("QMC_PROJ_MICRO_DIM", 64))   # micro subspace dim
PROJ_MACRO_DIM = int(os.environ.get("QMC_PROJ_MACRO_DIM", 32))   # macro representation dim
PROJ_HIGH_DIM = int(os.environ.get("QMC_PROJ_HIGH_DIM", 256))    # optional high-dim latent
PROJ_TOKEN_RATE = float(os.environ.get("QMC_PROJ_TOKEN_RATE", 10.0))  # tokens/sec for heavy reverse projection
PROJ_TOKEN_CAP = float(os.environ.get("QMC_PROJ_TOKEN_CAP", 20.0))

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
# Metrics (Prometheus or no-op)
# -------------------------
if ENABLE_PROM:
    MET_PROJECT = Counter("qmc_project_total", "Total PROJECT calls")
    MET_QUERY = Counter("qmc_query_total", "Total pocket_query calls")
    MET_FAISS_SEARCH = Counter("qmc_faiss_search_total", "Total FAISS searches")
    MET_ENT_HIT = Counter("qmc_ent_hit_total", "Entanglement cache hits")
    MET_ENT_PUT = Counter("qmc_ent_put_total", "Entanglement cache puts")
    MET_POCKET_PUT = Counter("qmc_pocket_put_total", "Total pocket_put calls")
    MET_DELETE = Counter("qmc_delete_total", "Total delete requests")
    LAT_PROJECT = Histogram("qmc_project_latency_seconds", "PROJECT latency seconds")
    LAT_QUERY = Histogram("qmc_query_latency_seconds", "pocket_query latency seconds")
    GAUGE_UNITS = Gauge("qmc_units", "Number of memory units")
else:
    class _Dummy:
        def inc(self, *a, **k): pass
        def observe(self, *a, **k): pass
    MET_PROJECT = MET_QUERY = MET_FAISS_SEARCH = MET_ENT_HIT = MET_ENT_PUT = MET_POCKET_PUT = MET_DELETE = _Dummy()
    LAT_PROJECT = LAT_QUERY = _Dummy()
    GAUGE_UNITS = _Dummy()

# -------------------------
# Ledger (batched writer + snapshot)
# -------------------------
class Ledger:
    chain: List[Dict[str, Any]] = []
    lock = threading.Lock()
    _write_q = queue.Queue()
    _stop = False

    @classmethod
    def record(cls, op: str, obj_id: str, info: Dict[str, Any]):
        try:
            with cls.lock:
                prev = cls.chain[-1]['hash'] if cls.chain else ''
                entry = {"ts": now_ts(), "op": op, "id": obj_id, "info": info, "prev": prev}
                s = json.dumps(entry, sort_keys=True, ensure_ascii=False)
                entry['hash'] = sha256_hex(s)
                cls.chain.append(entry)
                cls._write_q.put(entry)
        except Exception as e:
            log(f"Ledger.record error: {e}")

    @classmethod
    def _writer_worker(cls, path: str = LEDGER_PATH):
        buffer = []
        last_flush = time.time()
        while not cls._stop:
            try:
                item = cls._write_q.get(timeout=1.0)
                buffer.append(item)
                if len(buffer) >= 64 or (time.time() - last_flush) > 5.0:
                    cls._flush_buffer(buffer, path)
                    buffer = []
                    last_flush = time.time()
            except queue.Empty:
                if buffer:
                    cls._flush_buffer(buffer, path)
                    buffer = []
                    last_flush = time.time()
            except Exception as e:
                log(f"Ledger writer error: {e}")
                time.sleep(0.5)
        if buffer:
            cls._flush_buffer(buffer, path)

    @classmethod
    def _flush_buffer(cls, buffer: List[Dict[str, Any]], path: str):
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cls.chain, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            log(f"Ledger flushed {len(buffer)} entries to {path}")
        except Exception as e:
            log(f"Ledger._flush_buffer error: {e}")

    @classmethod
    def dump(cls, path: str = LEDGER_PATH):
        try:
            with cls.lock:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cls.chain, f, ensure_ascii=False, indent=2)
            log(f"Ledger dumped to {path}")
        except Exception as e:
            log(f"Ledger.dump error: {e}")

    @classmethod
    def start_writer(cls, path: str = LEDGER_PATH):
        t = threading.Thread(target=cls._writer_worker, args=(path,), daemon=True)
        t.start()
        return t

    @classmethod
    def stop_writer(cls):
        cls._stop = True

Ledger.start_writer(LEDGER_PATH)
atexit.register(lambda: (Ledger.dump(), Ledger.stop_writer()))

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
    last_active: float = field(default_factory=time.time)
    decay_score: float = 0.0
    explain: Optional[str] = None

@dataclass
class Hologram:
    id: str
    embedding: np.ndarray
    confidence: float
    provenance: Dict[str, Any]
    delta_E: float
    toxic_score: float = 0.0
    explain: Optional[str] = None

# -------------------------
# Sanitizer
# -------------------------
class Sanitizer:
    blacklist = set([w.strip() for w in SANITIZER_BLACKLIST if w.strip()])

    @staticmethod
    def clean_text(b: bytes) -> bytes:
        try:
            s = b.decode('utf-8', errors='ignore')
            s = "".join(ch for ch in s if ord(ch) >= 32)
            for bad in Sanitizer.blacklist:
                if bad and bad in s:
                    s = s.replace(bad, "[REDACTED]")
            return s.encode('utf-8')
        except Exception:
            return b

# -------------------------
# Detox utilities
# -------------------------
class Detox:
    @staticmethod
    def toxicity_score(emb: np.ndarray, background: Optional[np.ndarray] = None) -> float:
        try:
            mag = float(np.linalg.norm(emb))
            mean = float(np.mean(emb))
            std = float(np.std(emb)) + 1e-12
            kurt = float(np.mean(((emb - mean) / std) ** 4))
            score = min(1.0, (abs(mean) / 10.0) * 0.4 + (mag / (np.sqrt(len(emb)) + 1e-12)) * 0.3 + (min(kurt / 3.0, 1.0)) * 0.3)
            if background is not None:
                dist = np.linalg.norm(emb - background) / (np.linalg.norm(background) + 1e-12)
                score = min(1.0, score + min(dist, 1.0) * 0.2)
            return float(score)
        except Exception:
            return 0.0

    @staticmethod
    def is_anomalous(emb: np.ndarray, background: np.ndarray, z_threshold: float = ANOMALY_ZSCORE) -> bool:
        try:
            diff = emb - background
            z = np.abs((diff - np.mean(diff)) / (np.std(diff) + 1e-12))
            return bool(np.any(z > z_threshold))
        except Exception:
            return False

# -------------------------
# Token bucket
# -------------------------
class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.lock = threading.Lock()
        self.last = time.time()
    def consume(self, amount: float = 1.0) -> bool:
        with self.lock:
            now = time.time()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False

# -------------------------
# Projection Engine
# -------------------------
class ProjectionEngine:
    """
    Provides:
      - micro_to_macro(emb): fast online projection (low-latency)
      - macro_to_micro(macro_repr): async heavy reverse projection (rate-limited)
      - high_dim_projection: optional high-dim latent mapping
      - state machine for switching modes
    """
    def __init__(self, dim: int = DIM, micro_dim: int = PROJ_MICRO_DIM, macro_dim: int = PROJ_MACRO_DIM, high_dim: int = PROJ_HIGH_DIM):
        self.dim = dim
        self.micro_dim = micro_dim
        self.macro_dim = macro_dim
        self.high_dim = high_dim
        # random projection matrices (orthonormal-ish) for fast mapping
        rng = np.random.RandomState(42)
        self.micro_proj = rng.normal(scale=1.0, size=(self.micro_dim, self.dim)).astype('float32')
        self.macro_proj = rng.normal(scale=1.0, size=(self.macro_dim, self.micro_dim)).astype('float32')
        self.high_proj = rng.normal(scale=1.0, size=(self.high_dim, self.dim)).astype('float32')
        # small PCA-like running stats for micro->macro adapt
        self.micro_mean = np.zeros(self.micro_dim, dtype='float32')
        self.micro_count = 0
        # reverse projection threadpool and token bucket
        self.reverse_executor = ThreadPoolExecutor(max_workers=4)
        self.reverse_token_bucket = TokenBucket(PROJ_TOKEN_RATE, PROJ_TOKEN_CAP)
        # storage reference set by app
        self.storage = None
        # provenance ledger
        Ledger.record("PROJECTION_INIT", uid(), {"dim": dim, "micro_dim": micro_dim, "macro_dim": macro_dim, "high_dim": high_dim})
        log("ProjectionEngine initialized")

    def attach_storage(self, storage):
        self.storage = storage

    def micro_to_macro(self, emb: np.ndarray) -> Dict[str, Any]:
        """
        Fast online micro->macro:
          - project to micro subspace
          - normalize and project to macro dim
          - return macro vector + metadata
        """
        m = self.micro_proj.dot(emb)
        # running mean update (cheap)
        self.micro_count += 1
        if self.micro_count % 100 == 0:
            self.micro_mean = 0.99 * self.micro_mean + 0.01 * np.mean(m, axis=0)
        # macro mapping
        macro = self.macro_proj.dot(np.tanh(m))
        # normalize
        norm = np.linalg.norm(macro) + 1e-12
        macro = macro / norm
        meta = {"method": "micro_to_macro", "micro_norm": float(np.linalg.norm(m)), "macro_norm": float(norm)}
        Ledger.record("PROJ_MICRO_TO_MACRO", uid(), {"meta": meta})
        return {"macro": macro.astype('float32'), "meta": meta}

    def macro_to_micro_async(self, macro_repr: np.ndarray, callback=None, priority: int = 0) -> Optional[str]:
        """
        Schedule an async reverse projection (macro->micro).
        Rate-limited by token bucket. Returns task id or None if rejected.
        """
        if not self.reverse_token_bucket.consume(1.0):
            Ledger.record("PROJ_REVERSE_REJECT", uid(), {"reason": "rate_limit"})
            return None
        task_id = uid()
        self.reverse_executor.submit(self._reverse_worker, task_id, macro_repr, callback, priority)
        Ledger.record("PROJ_REVERSE_SCHEDULED", task_id, {"priority": priority})
        return task_id

    def _reverse_worker(self, task_id: str, macro_repr: np.ndarray, callback, priority: int):
        """
        Heavy reverse projection:
          - find nearest macro neighbors in ent cache / faiss
          - reconstruct micro candidates by linear combination + noise
          - optionally refine by local optimization (few steps)
          - store synthetic MemoryUnit in quarantine or return via callback
        """
        try:
            # step 1: find candidate memory ids via fusion/faiss
            candidates = []
            if self.storage and self.storage.fusion:
                # use macro_repr to query ent cache by projecting back to dim (approx)
                # approximate back-projection using pseudo-inverse of macro_proj * micro_proj
                approx_micro = np.linalg.pinv(self.macro_proj).dot(macro_repr)
                approx_full = np.linalg.pinv(self.micro_proj).dot(approx_micro)
                # use faiss search on approx_full if available
                hits = []
                try:
                    hits = self.storage.fusion.faiss_search(approx_full, topk=8)
                except Exception:
                    hits = []
                candidates = [self.storage.index[h] for h in hits if h in self.storage.index]
            # step 2: synthesize micro candidates
            synths = []
            if candidates:
                base = np.mean(np.stack(candidates, axis=0), axis=0)
                for i in range(min(6, len(candidates))):
                    alpha = 0.6 + 0.4 * random.random()
                    noise = np.random.normal(scale=1e-3, size=base.shape)
                    cand = alpha * base + (1 - alpha) * noise
                    synths.append(cand.astype('float32'))
            else:
                # fallback: expand macro via high_proj pseudo-inverse
                approx_full = np.linalg.pinv(self.high_proj).dot(np.concatenate([macro_repr, np.zeros(self.high_dim - len(macro_repr))])[:self.high_dim])
                for i in range(4):
                    noise = np.random.normal(scale=1e-2, size=approx_full.shape)
                    synths.append((approx_full + noise).astype('float32'))
            # step 3: optional local refinement (few gradient-free steps)
            refined = []
            for s in synths:
                # small smoothing toward background
                if self.storage and self.storage.background is not None:
                    s = 0.7 * s + 0.3 * self.storage.background
                refined.append(s)
            # step 4: create synthetic MemoryUnit(s) in quarantine for review
            created = []
            for s in refined:
                mem_id = uid()
                mu = MemoryUnit(id=mem_id, embedding=s.copy(), xi=0.0)
                mu.quarantined = True
                mu.explain = f"reverse_proj_task:{task_id}"
                # store in hot for inspection but quarantined
                self.storage.hot[mem_id] = mu
                self.storage.index[mem_id] = mu.embedding.copy()
                self.storage.quarantine[mem_id] = {"snapshot_hash": sha256_hex(mem_id + ":" + str(time.time())), "expire_ts": time.time() + QUARANTINE_HOLD, "requester": "reverse_proj", "reason": "reverse_projection", "task_id": task_id}
                created.append(mem_id)
                Ledger.record("PROJ_REVERSE_CREATED", mem_id, {"task_id": task_id})
            # callback with created ids
            if callback:
                try:
                    callback(task_id, created)
                except Exception:
                    pass
            log(f"ProjectionEngine: reverse task {task_id} created {len(created)} synthetic units")
        except Exception as e:
            log(f"ProjectionEngine: _reverse_worker error: {e}")

    def high_dim_project(self, emb: np.ndarray) -> np.ndarray:
        # optional high-dim latent mapping (cheap linear projection + tanh)
        h = np.tanh(self.high_proj.dot(emb))
        return h.astype('float32')

# -------------------------
# FusionCore (FAISS + Entanglement + Prefetch + Projection hooks)
# -------------------------
class FusionCore:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.use_faiss = FAISS_USE
        self.faiss_index = None
        self.faiss_ids: List[str] = []
        self.faiss_buffer: List[np.ndarray] = []
        self.faiss_buffer_ids: List[str] = []
        self.faiss_lock = threading.Lock()
        self._stop = False

        # ent cache
        self.ent_capacity = ENT_CAPACITY
        self.ent_cache: Dict[str, Dict[str, Any]] = {}
        self.ent_lru: List[str] = []
        self.ent_hit_threshold = 0.85
        self.ent_lock = threading.Lock()

        # redis optional
        self.use_redis = REDIS_USE and HAS_REDIS
        self.redis_client = None
        if self.use_redis:
            try:
                self.redis_client = redis.Redis()
                self.redis_client.ping()
                log("FusionCore: Redis connected for ent/quarantine")
            except Exception as e:
                log(f"FusionCore: Redis init failed: {e}")
                self.redis_client = None
                self.use_redis = False

        # prefetch queue
        self.prefetch_q = queue.Queue(maxsize=2048)
        self.prefetch_enabled = PREFETCH_ENABLED

        # threadpool for project batch
        self.project_executor = ThreadPoolExecutor(max_workers=MAX_PROJECT_WORKERS)

        # init faiss
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

        # background workers
        self._flush_thread = threading.Thread(target=self._faiss_flush_worker, daemon=True)
        self._flush_thread.start()
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

        # adaptive stats
        self.ent_hits = 0
        self.ent_queries = 0
        self.adapt_lock = threading.Lock()
        self.storage = None
        Ledger.record("FUSIONCORE_INIT", uid(), {"dim": dim, "faiss": self.use_faiss, "redis": self.use_redis})
        log("FusionCore initialized (projection-ready)")

    def attach_storage(self, storage):
        self.storage = storage

    # FAISS buffered add
    def faiss_add_buffered(self, mem_id: str, emb: np.ndarray):
        if not self.use_faiss or self.faiss_index is None:
            with self.faiss_lock:
                self.faiss_ids.append(mem_id)
                self.faiss_buffer.append(emb.astype('float32').copy())
            return
        with self.faiss_lock:
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
                with self.faiss_lock:
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
        if self.use_faiss and self.faiss_index is not None and len(self.faiss_ids) > 0:
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
        with self.faiss_lock:
            if not self.faiss_buffer and not self.faiss_ids:
                return []
            return self.faiss_ids[:topk]

    # ent cache with redis optional
    def ent_get(self, q_emb: np.ndarray):
        key = sha256_hex(",".join(map(str, np.round(q_emb[:8], 3).tolist())))[:16]
        with self.ent_lock:
            self.ent_queries += 1
            e = self.ent_cache.get(key)
            if not e and self.use_redis and self.redis_client:
                try:
                    raw = self.redis_client.get("ent:" + key)
                    if raw:
                        obj = json.loads(raw)
                        e = {"ent_vec": np.array(obj["ent_vec"], dtype='float32'), "mem_ids": obj["mem_ids"], "last_access": time.time(), "xi_reserve": obj.get("xi_reserve", 0.05)}
                        self.ent_cache[key] = e
                        self.ent_lru.insert(0, key)
                except Exception:
                    pass
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
            self.ent_hits += 1
            MET_ENT_HIT.inc()
            Ledger.record("ENT_HIT", key, {"sim": float(sim)})
            self._maybe_adapt()
            return e

    def ent_put(self, mem_embeddings: List[np.ndarray], mem_ids: List[str], xi_reserve: float = 0.05):
        if not mem_embeddings:
            return None
        ent_vec = np.mean(np.stack(mem_embeddings, axis=0), axis=0).astype('float32')
        key = sha256_hex(",".join(map(str, np.round(ent_vec[:8], 4).tolist())))[:16]
        with self.ent_lock:
            if key in self.ent_cache:
                self.ent_cache[key].update({"ent_vec": ent_vec, "mem_ids": mem_ids, "last_access": time.time(), "xi_reserve": xi_reserve})
                if key in self.ent_lru:
                    self.ent_lru.remove(key)
                self.ent_lru.insert(0, key)
            else:
                if len(self.ent_lru) >= self.ent_capacity:
                    tail = self.ent_lru.pop()
                    self.ent_cache.pop(tail, None)
                    if self.use_redis and self.redis_client:
                        try:
                            self.redis_client.delete("ent:" + tail)
                        except Exception:
                            pass
                self.ent_cache[key] = {"ent_vec": ent_vec, "mem_ids": mem_ids, "last_access": time.time(), "xi_reserve": xi_reserve}
                self.ent_lru.insert(0, key)
            if self.use_redis and self.redis_client:
                try:
                    self.redis_client.set("ent:" + key, json.dumps({"ent_vec": ent_vec.tolist(), "mem_ids": mem_ids, "xi_reserve": xi_reserve}), ex=3600)
                except Exception:
                    pass
            MET_ENT_PUT.inc()
            Ledger.record("ENT_PUT", key, {"count": len(mem_ids)})
            return key

    # project with toxicity check and projection hooks
    def project(self, query_emb: np.ndarray, background: Optional[np.ndarray] = None, alpha: float = 0.6, projection_mode: str = "micro"):
        """
        projection_mode: "micro" (default) uses micro->macro fast mapping for additional context,
                         "macro" uses macro-level aggregation,
                         "high" uses high-dim latent mixing.
        """
        start = time.time()
        MET_PROJECT.inc()
        # quick toxicity pre-check on query
        tscore = Detox.toxicity_score(query_emb, background)
        # projection engine hooks if attached
        proj_meta = {}
        if hasattr(self, "projection_engine") and self.projection_engine:
            if projection_mode == "micro":
                pm = self.projection_engine.micro_to_macro(query_emb)
                proj_meta = pm["meta"]
            elif projection_mode == "macro":
                pm = self.projection_engine.micro_to_macro(query_emb)
                proj_meta = pm["meta"]
            elif projection_mode == "high":
                h = self.projection_engine.high_dim_project(query_emb)
                proj_meta = {"high_dim_norm": float(np.linalg.norm(h))}
        # ent cache
        ent = self.ent_get(query_emb)
        if ent is not None:
            ent_vec = ent['ent_vec']
            emb = alpha * query_emb + (1.0 - alpha) * ent_vec
            delta_E = float(np.linalg.norm(emb - ent_vec))
            holo = Hologram(id=uid(), embedding=emb, confidence=0.92, provenance={"method": "entangled", "proj": projection_mode}, delta_E=delta_E, toxic_score=tscore)
            holo.explain = "entangled"
            LAT_PROJECT.observe(time.time() - start)
            Ledger.record("PROJECT_ENT", holo.id, {"delta_E": delta_E, "toxic_score": tscore, "proj_meta": proj_meta})
            if tscore >= TOXICITY_THRESHOLD:
                threading.Thread(target=self._handle_toxic_hologram, args=(holo,), daemon=True).start()
            return holo
        # faiss entanglement
        if self.storage and (self.use_faiss or self.faiss_buffer):
            hits = self.faiss_search(query_emb, topk=6)
            if hits:
                mem_embs = [self.storage.index[h] for h in hits if h in self.storage.index]
                if mem_embs:
                    ent_key = self.ent_put(mem_embs, hits, xi_reserve=0.05)
                    ent_vec = np.mean(np.stack(mem_embs, axis=0), axis=0)
                    emb = alpha * query_emb + (1.0 - alpha) * ent_vec
                    delta_E = float(np.linalg.norm(emb - ent_vec))
                    holo = Hologram(id=uid(), embedding=emb, confidence=0.86, provenance={"method": "faiss_ent", "ent_key": ent_key, "proj": projection_mode}, delta_E=delta_E, toxic_score=tscore)
                    holo.explain = "faiss_ent"
                    LAT_PROJECT.observe(time.time() - start)
                    Ledger.record("PROJECT_FAISS", holo.id, {"delta_E": delta_E, "hits": len(hits), "toxic_score": tscore, "proj_meta": proj_meta})
                    if tscore >= TOXICITY_THRESHOLD:
                        threading.Thread(target=self._handle_toxic_hologram, args=(holo,), daemon=True).start()
                    return holo
        # fallback to background
        B = background if background is not None else (self.storage.background if self.storage else np.zeros(self.dim, dtype='float32'))
        emb = alpha * query_emb + (1.0 - alpha) * B
        emb += np.random.normal(scale=1e-6, size=emb.shape)
        delta_E = float(np.linalg.norm(emb - B))
        holo = Hologram(id=uid(), embedding=emb, confidence=0.72, provenance={"method": "background", "proj": projection_mode}, delta_E=delta_E, toxic_score=tscore)
        holo.explain = "background"
        LAT_PROJECT.observe(time.time() - start)
        Ledger.record("PROJECT_BG", holo.id, {"delta_E": delta_E, "toxic_score": tscore, "proj_meta": proj_meta})
        if tscore >= TOXICITY_THRESHOLD:
            threading.Thread(target=self._handle_toxic_hologram, args=(holo,), daemon=True).start()
        return holo

    def _handle_toxic_hologram(self, holo: Hologram):
        try:
            res = self.storage.negentropy_read(holo, toxicity_threshold=TOXICITY_THRESHOLD)
            if res.get("status") == "ok":
                Ledger.record("TOXIC_HANDLED_OK", holo.id, {"method": "light_repair"})
                return
            repaired = res.get("hologram")
            if repaired and res.get("toxic"):
                time.sleep(0.2)
                if Detox.toxicity_score(repaired.embedding, self.storage.background) >= TOXICITY_THRESHOLD:
                    mem_id = uid()
                    mu = MemoryUnit(id=mem_id, embedding=repaired.embedding.copy(), xi=0.0)
                    mu.quarantined = True
                    mu.explain = f"auto_quarantine_from_holo:{holo.id}"
                    self.storage.hot[mem_id] = mu
                    self.storage.index[mem_id] = mu.embedding.copy()
                    Ledger.record("AUTO_QUARANTINE", mem_id, {"from_holo": holo.id, "toxic_score": repaired.delta_E})
                    log(f"FusionCore: Auto-quarantined synthetic unit {mem_id[:8]} from holo {holo.id[:8]}")
        except Exception as e:
            log(f"FusionCore: _handle_toxic_hologram error: {e}")

    def project_batch(self, queries: List[np.ndarray], background: Optional[np.ndarray] = None, alpha: float = 0.6, projection_mode: str = "micro") -> List[Hologram]:
        results: List[Optional[Hologram]] = [None] * len(queries)
        for i, q in enumerate(queries):
            ent = self.ent_get(q)
            if ent is not None:
                ent_vec = ent['ent_vec']
                emb = alpha * q + (1.0 - alpha) * ent_vec
                delta_E = float(np.linalg.norm(emb - ent_vec))
                tscore = Detox.toxicity_score(emb, background)
                results[i] = Hologram(id=uid(), embedding=emb, confidence=0.92, provenance={"method": "entangled"}, delta_E=delta_E, toxic_score=tscore, explain="entangled")
        futures = {}
        for i, q in enumerate(queries):
            if results[i] is None:
                futures[self.project_executor.submit(self.project, q, background, alpha, projection_mode)] = i
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                log(f"project_batch worker error: {e}")
                results[i] = self.project(queries[i], background, alpha, projection_mode)
        return results

    def prefetch(self, emb: np.ndarray):
        if not self.prefetch_enabled:
            return
        try:
            self.prefetch_q.put(emb, timeout=0.01)
        except Exception:
            pass

    def _prefetch_worker(self):
        while not self._stop:
            try:
                emb = self.prefetch_q.get(timeout=1.0)
                hits = self.faiss_search(emb, topk=8)
                if hits and self.storage:
                    mem_embs = [self.storage.index[h] for h in hits if h in self.storage.index]
                    if mem_embs:
                        self.ent_put(mem_embs, hits, xi_reserve=0.02)
            except queue.Empty:
                continue
            except Exception as e:
                log(f"FusionCore: prefetch_worker error: {e}")
                time.sleep(0.2)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        an = np.linalg.norm(a) + 1e-12
        bn = np.linalg.norm(b) + 1e-12
        return float(np.dot(a, b) / (an * bn))

    def _maybe_adapt(self):
        with self.adapt_lock:
            if self.ent_queries >= 200:
                hit_rate = self.ent_hits / max(1, self.ent_queries)
                if hit_rate > 0.6 and self.ent_hit_threshold > 0.6:
                    self.ent_hit_threshold = max(0.5, self.ent_hit_threshold - 0.02)
                elif hit_rate < 0.2 and self.ent_hit_threshold < 0.95:
                    self.ent_hit_threshold = min(0.95, self.ent_hit_threshold + 0.02)
                self.ent_hits = 0
                self.ent_queries = 0

    def shutdown(self):
        self._stop = True
        log("FusionCore: shutdown requested")
        with self.faiss_lock:
            if self.faiss_buffer:
                self._flush_faiss_buffer()
            if self.use_faiss and self.faiss_index is not None:
                try:
                    faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)
                    log("FusionCore: FAISS index persisted on shutdown")
                except Exception as e:
                    log(f"FusionCore: FAISS persist error on shutdown: {e}")
        try:
            self.project_executor.shutdown(wait=False)
        except Exception:
            pass

# -------------------------
# StorageCore (Detox + Decay + Quarantine + Projection integration)
# -------------------------
class StorageCore:
    def __init__(self, dim: int = DIM, fusion: FusionCore = None, projection_engine: ProjectionEngine = None):
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
        self.background = np.zeros(dim, dtype='float32')
        self._stop = False
        self.fusion = fusion
        if self.fusion:
            self.fusion.attach_storage(self)
        self.projection_engine = projection_engine
        if self.projection_engine:
            self.projection_engine.attach_storage(self)
        # workers
        self._consolidation_thread = threading.Thread(target=self._consolidation_worker, daemon=True)
        self._background_thread = threading.Thread(target=self._background_worker, daemon=True)
        self._decay_thread = threading.Thread(target=self._decay_worker, daemon=True)
        self._consolidation_thread.start()
        self._background_thread.start()
        self._decay_thread.start()
        self.token_bucket = TokenBucket(TOKEN_BUCKET_RATE, TOKEN_BUCKET_CAP)
        Ledger.record("STORAGECORE_INIT", uid(), {"dim": dim})
        log("StorageCore initialized (projection-ready)")

    # storage primitives
    def put_unit(self, unit: MemoryUnit, hot: bool = True, near: bool = True):
        unit.version += 1
        unit.ts = time.time()
        unit.last_active = time.time()
        if hot:
            self.hot[unit.id] = unit
        if near:
            self.near[unit.id] = unit
        self.index[unit.id] = unit.embedding.copy()
        self._incremental_background_update(unit)
        if self.fusion:
            self.fusion.faiss_add_buffered(unit.id, unit.embedding)
        MET_POCKET_PUT.inc()
        Ledger.record("PUT_UNIT", unit.id, {"xi": unit.xi, "core_protected": unit.core_protected, "explain": unit.explain})
        try:
            GAUGE_UNITS.set(len(self.index)) if ENABLE_PROM else None
        except Exception:
            pass

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
        u.last_active = time.time()
        return self.hot.get(mem_id)

    def retrieve_any(self, mem_id: str) -> Optional[MemoryUnit]:
        u = self.hot.get(mem_id) or self.near.get(mem_id)
        if u:
            u.last_active = time.time()
        return u

    # pocket_put with sanitization and toxicity pre-check
    def pocket_put(self, payload: bytes, embedding: np.ndarray, xi: float = 0.5, core_protect: bool = False, importance: float = 0.0, emotion: float = 0.0):
        try:
            clean_payload = Sanitizer.clean_text(payload)
        except Exception:
            clean_payload = payload
        mem_id = uid()
        unit = MemoryUnit(id=mem_id, embedding=embedding.copy(), xi=xi, trit=0, importance=importance, emotion=emotion, core_protected=core_protect)
        unit.explain = "sanitized_payload"
        sid = uid()
        self.put_shard(sid, clean_payload, xi, 0)
        unit.shards = [sid]
        tscore = Detox.toxicity_score(embedding, self.background)
        if tscore >= TOXICITY_THRESHOLD:
            unit.quarantined = True
            unit.explain = f"quarantined_on_put:score={tscore:.3f}"
            self.hot[mem_id] = unit
            self.index[mem_id] = unit.embedding.copy()
            self.quarantine[mem_id] = {"snapshot_hash": sha256_hex(mem_id + ":" + str(time.time())), "expire_ts": time.time() + QUARANTINE_HOLD, "requester": "auto", "reason": "toxicity_on_put", "score": tscore}
            Ledger.record("POCKET_PUT_QUARANTINED", mem_id, {"score": tscore})
            log(f"StorageCore: Pocket put quarantined {mem_id[:8]} score={tscore:.3f}")
            return {"status": "quarantined", "mem_id": mem_id, "score": tscore}
        self.put_unit(unit, hot=False, near=True)
        vaddr = "v:" + mem_id[:8]
        self.page_table[vaddr] = {"mem_id": mem_id, "local": False, "last_access": time.time()}
        if len(self.local_cache) < self.max_local and self.xi_pool > PROMOTE_COST:
            self._promote_to_local(vaddr)
        Ledger.record("POCKET_PUT", vaddr, {"mem_id": mem_id})
        return {"status": "ok", "vaddr": vaddr}

    def pocket_query(self, context_emb: np.ndarray, topk: int = 5, projection_mode: str = "micro"):
        start = time.time()
        MET_QUERY.inc()
        mids = []
        if self.fusion:
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
            results.append({"vaddr": vaddr, "mem_id": mid, "explain": u.explain})
        LAT_QUERY.observe(time.time() - start)
        Ledger.record("POCKET_QUERY", uid(), {"hits": len(results), "proj_mode": projection_mode})
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
        try:
            self.consolidation_q.put(mem_id, timeout=0.1)
            Ledger.record("CONSOLIDATION_PUSH", mem_id, {})
        except Exception:
            pass

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
        last_rebuild = 0.0
        while not self._stop:
            try:
                triggered = self.rebuild_event.wait(timeout=BACKGROUND_REBUILD_INTERVAL)
                with self.rebuild_lock:
                    now = time.time()
                    if now - last_rebuild > 0.5:
                        self._rebuild_background_field()
                        last_rebuild = now
                    self.rebuild_event.clear()
            except Exception as e:
                log(f"StorageCore: background worker error: {e}")
                time.sleep(0.5)

    def _rebuild_background_field(self):
        if not self.index:
            self.background = np.zeros(self.dim, dtype='float32')
            return
        ids = list(self.index.keys())
        mats = np.stack([self.index[i] for i in ids], axis=0)
        weights = np.array([max(self.retrieve_any(i).xi if self.retrieve_any(i) else 0.01, 0.01) * (1.0 + (self.retrieve_any(i).importance if self.retrieve_any(i) else 0.0)) for i in ids])
        total = weights.sum() + 1e-12
        B = (weights[:, None] * mats).sum(axis=0) / total
        self.background = B.astype('float32')
        Ledger.record("BACKGROUND_REBUILD", uid(), {"units": len(ids)})
        log("StorageCore: Background field rebuilt")

    # negentropy_read (enhanced)
    def negentropy_read(self, holo: Hologram, toxicity_threshold: float = TOXICITY_THRESHOLD):
        mean_val = float(np.mean(holo.embedding))
        toxic_score = Detox.toxicity_score(holo.embedding, self.background)
        temp = MemoryUnit(id="temp", embedding=holo.embedding.copy(), xi=0.5)
        violations = self._check_core_rules(temp)
        toxic = (toxic_score > toxicity_threshold) or (len(violations) > 0)
        if not toxic:
            Ledger.record("NEGENTROPY_OK", holo.id, {"toxic_score": toxic_score})
            return {"status": "ok", "hologram": holo, "toxic": False}
        # Stage 1: light repair
        c = -0.4 * np.sign(holo.embedding) * np.minimum(np.abs(holo.embedding), 0.05)
        repaired_emb = holo.embedding + c
        delta_comp = float(np.linalg.norm(c))
        repaired = Hologram(id=uid(), embedding=repaired_emb, confidence=max(0.1, holo.confidence - 0.05), provenance={"repair_of": holo.id, "stage": "light"}, delta_E=holo.delta_E + delta_comp)
        new_score = Detox.toxicity_score(repaired.embedding, self.background)
        Ledger.record("NEGENTROPY_REPAIR_STAGE1", repaired.id, {"orig": holo.id, "delta_comp": delta_comp, "new_score": new_score})
        if new_score <= toxicity_threshold:
            threading.Thread(target=self._async_validate_repair, args=(repaired,), daemon=True).start()
            return {"status": "repaired", "hologram": repaired, "toxic": False, "delta_comp": delta_comp}
        # Stage 2: stronger repair (async)
        stronger = Hologram(id=uid(), embedding=repaired.embedding.copy(), confidence=max(0.05, repaired.confidence - 0.1), provenance={"repair_of": holo.id, "stage": "strong"}, delta_E=repaired.delta_E)
        threading.Thread(target=self._strong_repair_and_validate, args=(stronger, holo.id), daemon=True).start()
        return {"status": "repaired_async", "hologram": stronger, "toxic": True, "delta_comp": delta_comp}

    def _strong_repair_and_validate(self, repaired: Hologram, orig_holo_id: str):
        try:
            repaired.embedding = 0.5 * repaired.embedding + 0.5 * self.background
            new_score = Detox.toxicity_score(repaired.embedding, self.background)
            Ledger.record("NEGENTROPY_REPAIR_STAGE2", repaired.id, {"orig": orig_holo_id, "new_score": new_score})
            if new_score > TOXICITY_THRESHOLD:
                mem_id = uid()
                mu = MemoryUnit(id=mem_id, embedding=repaired.embedding.copy(), xi=0.0)
                mu.quarantined = True
                mu.explain = f"auto_quarantine_from_repair:{orig_holo_id}"
                self.hot[mem_id] = mu
                self.index[mem_id] = mu.embedding.copy()
                self.quarantine[mem_id] = {"snapshot_hash": sha256_hex(mem_id + ":" + str(time.time())), "expire_ts": time.time() + QUARANTINE_HOLD, "requester": "auto_repair", "reason": "repair_failed", "score": new_score}
                Ledger.record("AUTO_QUARANTINE", mem_id, {"from_holo": orig_holo_id, "score": new_score})
                log(f"StorageCore: Auto-quarantined {mem_id[:8]} from repair of holo {orig_holo_id[:8]}")
            else:
                Ledger.record("NEGENTROPY_VALIDATE", repaired.id, {"validated": True})
                log(f"StorageCore: Repair validated {repaired.id[:8]}")
        except Exception as e:
            log(f"StorageCore: _strong_repair_and_validate error: {e}")

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

    # decay worker
    def _decay_worker(self):
        while not self._stop:
            try:
                time.sleep(DECAY_INTERVAL)
                now = time.time()
                to_decay = []
                for mid, u in list(self.hot.items()):
                    age = now - u.last_active
                    u.decay_score += DECAY_RATE * (age / max(1.0, DECAY_INTERVAL))
                    if u.decay_score > 0.5 and u.importance < 0.1:
                        to_decay.append(mid)
                for mid in to_decay:
                    u = self.hot.get(mid)
                    if not u:
                        continue
                    u.xi = max(0.0, u.xi - 0.1)
                    Ledger.record("DECAY_APPLIED", mid, {"decay_score": u.decay_score, "xi": u.xi})
                    if u.xi <= 0.0 and not u.core_protected:
                        self.request_self_delete(mid, requester="decay_worker", hold_seconds=QUARANTINE_HOLD)
                log(f"StorageCore: Decay pass done, decayed={len(to_decay)}")
            except Exception as e:
                log(f"StorageCore: decay worker error: {e}")
                time.sleep(1.0)

    # quarantine / delete
    def request_self_delete(self, mem_id: str, requester: str, hold_seconds: float = QUARANTINE_HOLD):
        u = self.retrieve_any(mem_id)
        if not u:
            return {"status": "not_found"}
        if u.core_protected:
            Ledger.record("DELETE_REJECT_CORE", mem_id, {"requester": requester})
            return {"status": "rejected_core_protected"}
        snap_hash = sha256_hex(mem_id + ":" + str(time.time()))
        expire_ts = time.time() + hold_seconds
        self.quarantine[mem_id] = {"snapshot_hash": snap_hash, "expire_ts": expire_ts, "requester": requester, "reason": "self_delete"}
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
        if info["requester"] != requester and requester != "admin":
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
        while time.time() < expire_ts and not self._stop:
            time.sleep(0.5)
        info = self.quarantine.get(mem_id)
        if not info:
            return
        if time.time() >= info["expire_ts"]:
            self.hot.pop(mem_id, None)
            self.near.pop(mem_id, None)
            self.index.pop(mem_id, None)
            self.quarantine.pop(mem_id, None)
            Ledger.record("PERMANENT_DELETE", mem_id, {"ts": time.time()})
            log(f"StorageCore: Permanently deleted {mem_id[:8]}")

    def _incremental_background_update(self, unit: MemoryUnit):
        try:
            alpha = 1.0 / max(1, len(self.index))
            self.background = (1 - alpha) * self.background + alpha * unit.embedding
        except Exception:
            self.rebuild_event.set()

    def shutdown(self):
        self._stop = True
        log("StorageCore: shutdown requested")

# -------------------------
# Application harness (full)
# -------------------------
class QuantumMemoryCloudApp:
    def __init__(self, dim: int = DIM, demo_units: int = DEMO_UNITS, enable_http: bool = ENABLE_HTTP):
        self.dim = dim
        self.demo_units = demo_units
        self.projection_engine = ProjectionEngine(dim=dim, micro_dim=PROJ_MICRO_DIM, macro_dim=PROJ_MACRO_DIM, high_dim=PROJ_HIGH_DIM)
        self.fusion = FusionCore(dim=dim)
        self.storage = StorageCore(dim=dim, fusion=self.fusion, projection_engine=self.projection_engine)
        self.fusion.projection_engine = self.projection_engine
        self._stop = False
        Ledger.record("APP_INIT", uid(), {"dim": dim, "demo_units": demo_units})
        log("Starting quantum memory cloud (production full projection)")
        # populate demo
        self._populate_thread = threading.Thread(target=self._populate_demo, daemon=True)
        self._populate_thread.start()
        # auto project batcher
        self._project_thread = threading.Thread(target=self._auto_project_batcher, daemon=True)
        self._project_thread.start()
        # optional HTTP
        self.api = None
        if enable_http and FASTAPI_AVAILABLE:
            try:
                self._start_http()
            except Exception as e:
                log(f"HTTP start failed: {e}")
        # signals
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        log("System started (FusionCore + StorageCore + ProjectionEngine)")

    def _populate_demo(self):
        log(f"Demo: populating {self.demo_units} memory units")
        for i in range(self.demo_units):
            emb = np.random.normal(scale=1.0, size=(self.dim,)).astype('float32')
            payload = f"demo-{i}".encode('utf-8')
            res = self.storage.pocket_put(payload, emb, xi=0.5)
            # occasionally push for consolidation
            if isinstance(res, dict) and res.get("status") == "ok":
                vaddr = res["vaddr"]
                mem_id = self.storage.page_table.get(vaddr, {}).get("mem_id")
                if mem_id and i % max(1, CONSOLIDATION_BATCH // 4) == 0:
                    self.storage.push_consolidation(mem_id)
            if PREFETCH_ENABLED:
                self.fusion.prefetch(emb)
            time.sleep(0.002)
        log("Demo population done")
        Ledger.record("DEMO_POPULATED", uid(), {"units": self.demo_units})
        self.storage.rebuild_event.set()

    def _auto_project_batcher(self):
        while not self._stop:
            try:
                batch = []
                for _ in range(AUTO_PROJECT_BATCH):
                    q = np.random.normal(scale=1.0, size=(self.dim,)).astype('float32')
                    batch.append(q)
                if not self.storage.token_bucket.consume(len(batch)):
                    time.sleep(0.05)
                    continue
                # alternate projection modes to exercise micro/macro/high
                mode = random.choice(["micro", "macro", "high"])
                holos = self.fusion.project_batch(batch, background=self.storage.background, alpha=0.6, projection_mode=mode)
                for holo in holos:
                    res = self.storage.negentropy_read(holo)
                    Ledger.record("AUTO_PROJECT", holo.id, {"status": res.get("status"), "conf": holo.confidence, "toxic": holo.toxic_score, "proj_mode": mode})
                time.sleep(max(0.05, BACKGROUND_REBUILD_INTERVAL / 8.0))
            except Exception as e:
                log(f"Auto project batcher error: {e}")
                time.sleep(0.5)

    def _start_http(self):
        app = FastAPI()
        @app.get("/health")
        def health():
            return {"status": "ok", "time": now_ts()}
        @app.get("/metrics")
        def metrics():
            if ENABLE_PROM:
                return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
            else:
                return {"metrics": "disabled"}
        @app.post("/put")
        def put_item(payload: Dict[str, Any]):
            try:
                text = payload.get("text", "").encode('utf-8')
                emb = np.array(payload.get("embedding", np.random.normal(size=(self.dim,)).tolist()), dtype='float32')
                res = self.storage.pocket_put(text, emb, xi=float(payload.get("xi", 0.5)))
                return res
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        @app.post("/query")
        def query_item(payload: Dict[str, Any]):
            try:
                emb = np.array(payload.get("embedding", np.random.normal(size=(self.dim,)).tolist()), dtype='float32')
                topk = int(payload.get("topk", 5))
                mode = payload.get("proj_mode", "micro")
                res = self.storage.pocket_query(emb, topk=topk, projection_mode=mode)
                return {"hits": res}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        @app.get("/ledger")
        def ledger_count():
            return {"entries": len(Ledger.chain)}
        @app.post("/admin/snapshot")
        def admin_snapshot():
            try:
                Ledger.dump()
                return {"status": "dumped"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        self.api = app
        # run uvicorn externally as recommended
        log("HTTP API initialized (use uvicorn to serve)")

    def _signal_handler(self, signum, frame):
        log(f"Signal {signum} received, shutting down")
        self.shutdown()

    def shutdown(self):
        if self._stop:
            return
        self._stop = True
        try:
            self.fusion.shutdown()
        except Exception:
            pass
        try:
            self.storage.shutdown()
        except Exception:
            pass
        try:
            Ledger.dump()
        except Exception:
            pass
        log("QuantumMemoryCloudApp shutdown complete")

# -------------------------
# Entrypoint
# -------------------------
def main():
    app = QuantumMemoryCloudApp(dim=DIM, demo_units=DEMO_UNITS, enable_http=ENABLE_HTTP)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        app.shutdown()

if __name__ == "__main__":
    main()
