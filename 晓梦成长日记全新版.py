# -*- coding: utf-8 -*-
"""
Xiaomeng — Single-Persona Soul Algorithm Enhanced with Quantum Entanglement Memory
================================================================================
This is the complete fusion of Xiaomeng algorithm with Quantum Entanglement Memory (QEM).

Key Features:
- Original Xiaomeng framework preserved (personality, emotions, social, sleep, etc.)
- Memory layer completely replaced with QEM system
- QEM provides advanced vector compression, clustering, and quantum-inspired memory organization
- Seamless integration through QEMMemory adapter
- All original functionality maintained while gaining quantum memory capabilities

Author: Fused Integration System
Python 3.8+
"""

# ==============================================================================
# 基础导入和配置
# ==============================================================================

import os
import time
import random
import json
import uuid
import threading
import queue
import math
import hashlib
import shutil
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# 可选加速支持
try:
    import numpy as np
except Exception:
    np = None

# ==============================================================================
# 基础配置和路径
# ==============================================================================

DEBUG = True
DISPLAY_NAME = "Xiaomeng"
BASE_DIR = os.path.abspath(".")
SHARD_DIR = os.path.join(BASE_DIR, "shards")
INDEX_FILE = os.path.join(BASE_DIR, "hybrid_index.json")

# QEM相关配置
QEM_DATA_DIR = os.path.join(BASE_DIR, "qem_cloud_data")
QEM_SHARD_DIR = os.path.join(QEM_DATA_DIR, "shards")
QEM_ENT_PATH = os.path.join(QEM_DATA_DIR, "ents.json")
QEM_SEED_PATH = os.path.join(QEM_DATA_DIR, "seeds.json")
QEM_LEDGER_PATH = os.path.join(QEM_DATA_DIR, "ledger.json")
QEM_COMP_LOG = os.path.join(QEM_DATA_DIR, "compression_log.json")
QEM_QUARANTINE_PATH = os.path.join(QEM_DATA_DIR, "quarantine.json")

# 原有配置路径
PERSONA_FILE = os.path.join(BASE_DIR, "persona_ledger.json")
PERSISTED_PERSONA = os.path.join(BASE_DIR, "persona_single.json")
SECURE_LOG_DIR = os.path.join(BASE_DIR, "securelogs")
LOCAL_KEY_PATH = os.path.join(BASE_DIR, "localkey.bin")
AUDIT_LOG = os.path.join(BASE_DIR, "audit_log.jsonl")

# 系统参数
VECTOR_DIM = int(os.environ.get("QEM_DIM", 64))
INITIAL_THINKER_COUNT = 12
TRAUMA_DECAY = 0.92

XI_POOL_MAX = 100
XI_POOL_WORK_COST = 5
XI_POOL_SLEEP_RECOVERY = 10
XI_POOL_LOW_THRESHOLD = 20

REFUSE_BASE_RATE = 0.36
FATIGUE_REFUSE_THRESHOLD = 0.5

DAY_START = 7
DAY_END = 19

SEED = 2026
random.seed(SEED)

# QEM系统参数
QEM_DEFAULT_SIM = float(os.environ.get("QEM_SIM", 0.70))
QEM_DEFAULT_ITERS = int(os.environ.get("QEM_ITERS", 4))
QEM_QUANT_BITS = int(os.environ.get("QEM_QUANT_BITS", 8))
QEM_MIN_GROUP = int(os.environ.get("QEM_MIN_GROUP", 2))
QEM_FREQ_ALPHA = float(os.environ.get("QEM_FREQ_ALPHA", 1.0))
QEM_FREQ_BETA = float(os.environ.get("QEM_FREQ_BETA", 1.0))
QEM_PAIR_SIM_FACTOR = float(os.environ.get("QEM_PAIR_SIM_FACTOR", 1.0))
QEM_SIM_MIN = float(os.environ.get("QEM_SIM_MIN", 0.35))
QEM_QUARANTINE_RETRY = int(os.environ.get("QEM_QUARANTINE_RETRY", 3))

# ==============================================================================
# 基础工具函数
# ==============================================================================

def ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

# 创建所有需要的目录
ensure_dir(SECURE_LOG_DIR)
ensure_dir(SHARD_DIR)
ensure_dir(QEM_DATA_DIR)
ensure_dir(QEM_SHARD_DIR)

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def uid(prefix: str = "") -> str:
    return prefix + str(uuid.uuid4())[:12]

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def load_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

def save_json(path: str, data: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def safe_write_json(path: str, obj: Any):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

# ==============================================================================
# 密钥和签名系统
# ==============================================================================

def load_or_create_local_key(path: str = LOCAL_KEY_PATH) -> bytes:
    if os.path.exists(path):
        try:
            return open(path, "rb").read()
        except Exception:
            pass
    k = os.urandom(32)
    try:
        with open(path, "wb") as f:
            f.write(k)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
    except Exception:
        pass
    return k

def generate_hmac_key() -> bytes:
    return hashlib.sha256(str(random.getrandbits(256)).encode()).digest()

LOCAL_KEY = load_or_create_local_key()
AUTONOMOUS_HMAC_KEY = generate_hmac_key()

def sign_event(event: Dict[str, Any]) -> str:
    try:
        data = json.dumps(event, ensure_ascii=False, sort_keys=True)
        sig = hashlib.sha256((data + str(AUTONOMOUS_HMAC_KEY)).encode()).hexdigest()
        return sig
    except Exception:
        return hashlib.sha256(str(event).encode()).hexdigest()

# ==============================================================================
# 事件记录系统（人格账本 + 安全日志）
# ==============================================================================

def persona_record_event(entity_id: str, event: Dict[str, Any], secure: bool = False):
    try:
        ev = dict(event)
        if DEBUG:
            ev["debug_time"] = now_ts()
            ev["debug_thread"] = threading.current_thread().name
            try:
                ev["debug_preview"] = (json.dumps(ev, ensure_ascii=False)[:800] + "...") \
                    if len(json.dumps(ev, ensure_ascii=False)) > 800 else json.dumps(ev, ensure_ascii=False)
            except Exception:
                ev["debug_preview"] = str(ev)[:800]
        
        ev["signature"] = sign_event(ev)
        
        ledger = load_json(PERSONA_FILE, {"entities": {}, "events": []})
        ledger["events"].append({"entity": entity_id, "time": now_ts(), "event": ev})
        
        ent = ledger["entities"].setdefault(entity_id, {"history": [], "traits": {}})
        ent["history"].append({"time": now_ts(), "event": ev})
        
        save_json(PERSONA_FILE, ledger)
        
        if secure:
            try:
                path = os.path.join(SECURE_LOG_DIR, f"event_{uid()}.log")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(ev, ensure_ascii=False))
            except Exception:
                pass
        
        if DEBUG:
            try:
                print(f"[DEBUG EVENT] {entity_id} | {ev.get('type','event')} | {ev.get('debug_preview', ev)}")
            except Exception:
                print(f"[DEBUG EVENT] {entity_id} | {ev.get('type','event')}")
                
    except Exception as e:
        try:
            with open(os.path.join(SECURE_LOG_DIR, "persona_record_error.log"), "a", encoding="utf-8") as f:
                f.write(f"{now_ts()} persona_record_event error: {repr(e)}\n")
        except Exception:
            pass

# ==============================================================================
# 核心状态：XiPool, TraumaManager, AgentState
# ==============================================================================

class XiPool:
    def __init__(self, max_pool: int = XI_POOL_MAX):
        self.max_pool = max_pool
        self.current = max_pool
        self._lock = threading.Lock()
    
    def consume(self, amount: int = XI_POOL_WORK_COST) -> bool:
        with self._lock:
            if self.current >= amount:
                self.current -= amount
                return True
            return False
    
    def recover(self, amount: int = XI_POOL_SLEEP_RECOVERY):
        with self._lock:
            self.current = min(self.max_pool, self.current + amount)
    
    def is_low(self) -> bool:
        with self._lock:
            return self.current < XI_POOL_LOW_THRESHOLD
    
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "current": self.current,
                "max": self.max_pool,
                "ratio": round(self.current / self.max_pool if self.max_pool else 0.0, 3)
            }

class TraumaManager:
    def __init__(self, decay_rate: float = TRAUMA_DECAY):
        self.traumas: List[Dict[str, Any]] = []
        self.decay_rate = decay_rate
        self._lock = threading.Lock()
    
    def add(self, trauma: Dict[str, Any]):
        with self._lock:
            trauma["id"] = uid()
            trauma["time"] = now_ts()
            trauma["severity"] = trauma.get("severity", 1.0)
            self.traumas.append(trauma)
            persona_record_event(DISPLAY_NAME, {"type": "trauma_added", "trauma_id": trauma["id"]}, secure=True)
    
    def decay(self):
        with self._lock:
            self.traumas = [t for t in self.traumas if random.random() > self.decay_rate * t.get("severity", 1.0)]
    
    def severity_score(self) -> float:
        with self._lock:
            if not self.traumas:
                return 0.0
            return min(1.0, sum(t.get("severity", 1.0) for t in self.traumas) / 10.0)

class AgentState:
    def __init__(self):
        self._lock = threading.Lock()
        self.state = "work"
        self.fatigue = 0.0
        self.mood = 0.5
        self.produced = 0
        self.sleep_lock = False
        self.sleep_count = 0
        self.refuse_count = 0
        self.cycles = 0
        self.xi_pool = XiPool()
        self.trauma_manager = TraumaManager()
    
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "fatigue": round(self.fatigue, 3),
                "mood": round(self.mood, 3),
                "produced": self.produced,
                "sleep_lock": self.sleep_lock,
                "sleep_count": self.sleep_count,
                "refuse_count": self.refuse_count,
                "xi_pool": self.xi_pool.snapshot(),
                "trauma_severity": round(self.trauma_manager.severity_score(), 3)
            }
    
    def request_sleep(self, reason: str = "") -> bool:
        with self._lock:
            req_id = sleep_queue.request(requester=DISPLAY_NAME, reason=reason)
            persona_record_event(DISPLAY_NAME, {
                "type": "sleep_request_accepted",
                "req_id": req_id,
                "reason": reason,
                "sleep_lock": self.sleep_lock,
                "fatigue": round(self.fatigue, 3)
            }, secure=True)
            return True
    
    def apply_sleep_cycle(self, sleep_recovery: float = 0.08, env_profile: Optional[str] = None, story_id: Optional[str] = None):
        with self._lock:
            old = self.fatigue
            self.fatigue = max(0.0, self.fatigue - sleep_recovery)
            self.mood = min(1.0, self.mood + 0.02)
            self.xi_pool.recover()
            self.trauma_manager.decay()
            
            # Memory consolidation hook using QEM
            try:
                qem_memory.create_shard([{
                    "question": "sleep_consolidation",
                    "fragment": json.dumps({
                        "time": now_ts(),
                        "old_fatigue": round(old, 3),
                        "new_fatigue": round(self.fatigue, 3),
                        "env_profile": env_profile,
                        "story_id": story_id
                    }, ensure_ascii=False),
                    "tags": ["sleep", "consolidation"]
                }])
            except Exception:
                pass
            
            self.sleep_lock = False
            persona_record_event(DISPLAY_NAME, {
                "type": "sleep_cycle",
                "old_fatigue": round(old, 3),
                "new_fatigue": round(self.fatigue, 3),
                "env_profile": env_profile,
                "story_id": story_id
            }, secure=True)
    
    def apply_work_cycle(self, produced: int = 1, fatigue_increase: float = 0.03):
        with self._lock:
            self.produced += produced
            self.fatigue = min(1.0, self.fatigue + fatigue_increase)
            self.mood = max(0.0, self.mood - 0.005)
            self.xi_pool.consume()
            persona_record_event(DISPLAY_NAME, {
                "type": "work_cycle",
                "produced": produced,
                "fatigue": round(self.fatigue, 3)
            }, secure=True)
    
    def should_refuse_work(self) -> bool:
        with self._lock:
            refuse_prob = REFUSE_BASE_RATE
            if self.fatigue > FATIGUE_REFUSE_THRESHOLD:
                refuse_prob += (self.fatigue - FATIGUE_REFUSE_THRESHOLD) * 0.5
            if self.xi_pool.is_low():
                refuse_prob += 0.2
            trauma_score = self.trauma_manager.severity_score()
            refuse_prob += trauma_score * 0.3
            if self.mood < 0.3:
                refuse_prob += 0.15
            refuse_prob = refuse_prob * (0.8 + random.random() * 0.4)
            return random.random() < min(refuse_prob, 0.9)

agent_state = AgentState()

# ==============================================================================
# 环境和时间系统
# ==============================================================================

class Environment:
    def __init__(self, start_ts: Optional[float] = None, timezone_offset_hours: int = 0, speed_factor: float = 1.0):
        self.start_real = time.time()
        self.start_sim = start_ts if start_ts is not None else time.time()
        self.timezone_offset = timezone_offset_hours
        self.speed_factor = speed_factor
        self._lock = threading.Lock()
    
    def now_sim(self) -> float:
        with self._lock:
            real_elapsed = time.time() - self.start_real
            sim_elapsed = real_elapsed * self.speed_factor
            return self.start_sim + sim_elapsed
    
    def current_hour(self) -> int:
        ts = self.now_sim() + self.timezone_offset * 3600
        return time.localtime(ts).tm_hour
    
    def is_day(self) -> bool:
        h = self.current_hour()
        return DAY_START <= h < DAY_END
    
    def current_season(self) -> str:
        month = time.localtime(self.now_sim()).tm_mon
        if month in (3, 4, 5):
            return "Spring"
        if month in (6, 7, 8):
            return "Summer"
        if month in (9, 10, 11):
            return "Autumn"
        return "Winter"
    
    def tick(self):
        return {
            "hour": self.current_hour(),
            "is_day": self.is_day(),
            "season": self.current_season(),
            "ts": now_ts()
        }

environment = Environment()

# ==============================================================================
# 梦境引擎
# ==============================================================================

class DreamEngine:
    def __init__(self):
        self.symbols = {
            "objects": [
                "afterglow", "sea of fire", "sweater", "milky way", "old photo", "night lamp",
                "river", "ashes", "echo", "window", "paper boat", "wind chime", "crescent",
                "tide", "breath", "temperature"
            ],
            "emotions": [
                "like an unfinished song", "whispering on the chest", "like the breath of the tide",
                "like the corner of an old envelope", "like a lamp in the night", 
                "like a forgotten scent", "like the warmth at fingertips", "like the smell of soil after rain"
            ]
        }
        self._lock = threading.Lock()
    
    def generate_dream(self, context: Optional[Dict[str, Any]] = None) -> str:
        with self._lock:
            depth = 3 + random.randint(0, 4)
            picks = random.sample(self.symbols["objects"], k=min(len(self.symbols["objects"]), depth + 2))
            weave = " · ".join(picks) + " — "
            parts = []
            for _ in range(depth):
                obj = random.choice(self.symbols["objects"])
                emotion = random.choice(self.symbols["emotions"])
                parts.append(f"{obj}, {emotion}")
            weave += " / ".join(parts)
            persona_record_event(DISPLAY_NAME, {"type": "dream_generated", "preview": weave}, secure=False)
            return weave
    
    def replay(self, fragments: List[Dict[str, Any]], mode: str = "rehearse", limit: int = 8) -> List[str]:
        with self._lock:
            picks = fragments[-limit:]
            outputs = []
            for p in picks:
                frag = p.get("fragment", "") if isinstance(p, dict) else str(p)
                if mode == "re-eval":
                    outputs.append(f"Counterfactual replay: {frag}")
                elif mode == "consolidate":
                    outputs.append(f"Consolidation: {frag}")
                else:
                    outputs.append(f"Rehearsal: {frag}")
            return outputs

dream_engine = DreamEngine()

# ==============================================================================
# 概念图和新颖性检测
# ==============================================================================

@dataclass
class Concept:
    id: str
    label: str
    prototypes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    weight: float = 1.0
    time: str = field(default_factory=now_ts)

class ConceptGraph:
    def __init__(self):
        self.nodes: Dict[str, Concept] = {}
        self.edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
    
    def add_concept(self, label: str, prototypes: Optional[List[str]] = None, tags: Optional[List[str]] = None) -> str:
        cid = uid()
        c = Concept(id=cid, label=label, prototypes=prototypes or [], tags=tags or [])
        self.nodes[cid] = c
        return cid
    
    def link(self, a: str, b: str, rel: str = "related", weight: float = 1.0):
        self.edges[(a, b)] = {"rel": rel, "weight": weight}
        self.edges[(b, a)] = {"rel": rel, "weight": weight}
    
    def align_concepts(self, text: str) -> List[Tuple[str, float]]:
        toks = text.lower().split()
        scores = []
        for cid, c in self.nodes.items():
            score = sum(1 for t in toks if t in c.label.lower() or any(t in p.lower() for p in c.prototypes))
            scores.append((c.label, score / (len(toks) or 1)))
        scores.sort(key=lambda x: -x[1])
        return scores[:5]

concept_graph = ConceptGraph()

# ==============================================================================
# 量子纠缠记忆系统 (QEM) - 完整实现
# ==============================================================================

# 数据模型
@dataclass
class EntNode:
    id: str
    vec: Optional[List[float]]
    shards: List[str]
    score: float = 1.0
    ts: float = field(default_factory=time.time)
    
    def to_dict(self):
        return {
            "id": self.id,
            "vec": self.vec,
            "shards": self.shards,
            "score": self.score,
            "ts": self.ts
        }
    
    @staticmethod
    def from_dict(d):
        return EntNode(
            id=d["id"],
            vec=d.get("vec"),
            shards=d.get("shards", []),
            score=d.get("score", 1.0),
            ts=d.get("ts", time.time())
        )

@dataclass
class SeedNode:
    id: str
    seed_vec: Optional[List[float]]
    members: List[str]
    diffs: Dict[str, List[int]] = field(default_factory=dict)
    quant_meta: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    
    def to_dict(self):
        return {
            "id": self.id,
            "seed_vec": self.seed_vec,
            "members": self.members,
            "diffs": self.diffs,
            "quant_meta": self.quant_meta,
            "ts": self.ts
        }
    
    @staticmethod
    def from_dict(d):
        return SeedNode(
            id=d["id"],
            seed_vec=d.get("seed_vec"),
            members=d.get("members", []),
            diffs=d.get("diffs", {}),
            quant_meta=d.get("quant_meta", {}),
            ts=d.get("ts", time.time())
        )

# 向量操作辅助函数
def _cosine(a, b):
    if not a or not b:
        return 0.0
    try:
        if np is not None:
            aa = np.array(a, dtype=float)
            bb = np.array(b, dtype=float)
            an = np.linalg.norm(aa) + 1e-12
            bn = np.linalg.norm(bb) + 1e-12
            return float(np.dot(aa, bb) / (an * bn))
    except Exception:
        pass
    
    m = min(len(a), len(b))
    dot = sum((a[i] * b[i]) for i in range(m))
    an = math.sqrt(sum(x*x for x in a)) + 1e-12
    bn = math.sqrt(sum(y*y for y in b)) + 1e-12
    return dot / (an * bn)

def mean_vec(vecs):
    if not vecs:
        return None
    try:
        if np is not None:
            arr = np.stack([np.array(v, dtype=float) for v in vecs], axis=0)
            return np.mean(arr, axis=0).tolist()
    except Exception:
        pass
    
    dim = max(len(v) for v in vecs)
    res = [0.0] * dim
    for v in vecs:
        for i, x in enumerate(v):
            res[i] += x
    n = len(vecs)
    return [x / n for x in res]

def vec_sub(a, b):
    if a is None or b is None:
        return None
    dim = max(len(a), len(b))
    res = []
    for i in range(dim):
        ai = a[i] if i < len(a) else 0.0
        bi = b[i] if i < len(b) else 0.0
        res.append(ai - bi)
    return res

def vec_add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    dim = max(len(a), len(b))
    res = []
    for i in range(dim):
        ai = a[i] if i < len(a) else 0.0
        bi = b[i] if i < len(b) else 0.0
        res.append(ai + bi)
    return res

# 量化辅助函数
def quantize_list(vecs: List[List[float]], bits: int = QEM_QUANT_BITS):
    flat = [x for v in vecs for x in v] if vecs else []
    if not flat:
        return [], {}
    mn = min(flat)
    mx = max(flat)
    if mn == mx:
        q = [[0] * len(vecs[0]) for _ in vecs]
        return q, {"min": mn, "max": mx, "bits": bits}
    
    levels = (1 << bits) - 1
    meta = {"min": mn, "max": mx, "bits": bits}
    qvecs = []
    for v in vecs:
        qv = [int(round((x - mn) / (mx - mn) * levels)) for x in v]
        qvecs.append(qv)
    return qvecs, meta

def dequantize(qvec, meta):
    mn = meta.get("min", 0.0)
    mx = meta.get("max", 0.0)
    bits = meta.get("bits", QEM_QUANT_BITS)
    levels = (1 << bits) - 1
    if levels == 0:
        return [mn for _ in qvec]
    return [mn + (x / levels) * (mx - mn) for x in qvec]

# QEM核心类
class EntRegistry:
    def __init__(self, path=QEM_ENT_PATH):
        self.path = path
        self.lock = threading.RLock()
        self.nodes: Dict[str, EntNode] = {}
        self._load()
    
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for nid, nd in data.get("nodes", {}).items():
                        self.nodes[nid] = EntNode.from_dict(nd)
            except Exception:
                self.nodes = {}
    
    def save(self):
        with self.lock:
            data = {"nodes": {nid: n.to_dict() for nid, n in self.nodes.items()}}
            safe_write_json(self.path, data)
    
    def register(self, node: EntNode):
        with self.lock:
            self.nodes[node.id] = node
            try:
                safe_write_json(self.path, {"nodes": {nid: n.to_dict() for nid, n in self.nodes.items()}})
            except Exception:
                pass

class SeedIndex:
    def __init__(self, path=QEM_SEED_PATH):
        self.path = path
        self.lock = threading.RLock()
        self.seeds: Dict[str, SeedNode] = {}
        self._load()
    
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for sid, sd in data.get("seeds", {}).items():
                        self.seeds[sid] = SeedNode.from_dict(sd)
            except Exception:
                self.seeds = {}
    
    def save(self):
        with self.lock:
            data = {"seeds": {sid: s.to_dict() for sid, s in self.seeds.items()}}
            safe_write_json(self.path, data)
    
    def register(self, seed: SeedNode):
        with self.lock:
            self.seeds[seed.id] = seed
            try:
                safe_write_json(self.path, {"seeds": {sid: s.to_dict() for sid, s in self.seeds.items()}})
            except Exception:
                pass

# 频率存储
class FrequencyStore:
    def __init__(self, path=os.path.join(QEM_DATA_DIR, "freq.json")):
        self.path = path
        self.lock = threading.RLock()
        self.counts: Dict[str, int] = {}
        self._load()
    
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.counts = json.load(f)
            except Exception:
                self.counts = {}
    
    def inc(self, k: str, d: int = 1):
        with self.lock:
            self.counts[k] = self.counts.get(k, 0) + d
            if self.counts[k] % 50 == 0:
                safe_write_json(self.path, self.counts)
    
    def get(self, k: str) -> int:
        with self.lock:
            return self.counts.get(k, 0)

freq_store = FrequencyStore()

# QEM压缩器
class QEMCompressor:
    def __init__(self, registry: EntRegistry, seed_index: SeedIndex):
        self.registry = registry
        self.seed_index = seed_index
    
    def greedy_cluster(self, nodes: List[EntNode], sim_thresh: float, min_group: int):
        groups = []
        used = set()
        for i, a in enumerate(nodes):
            if a.id in used:
                continue
            group = [a]
            used.add(a.id)
            for b in nodes[i+1:]:
                if b.id in used:
                    continue
                try:
                    if _cosine(a.vec, b.vec) >= sim_thresh:
                        group.append(b)
                        used.add(b.id)
                except Exception:
                    continue
            if len(group) >= min_group:
                groups.append(group)
        return groups
    
    def build_seeds(self, sim_thresh: float = QEM_DEFAULT_SIM, min_group: int = QEM_MIN_GROUP, quant_bits: int = QEM_QUANT_BITS):
        nodes = list(self.registry.nodes.values())
        if not nodes:
            return {"created": 0}
        
        groups = self.greedy_cluster(nodes, sim_thresh, min_group)
        created = 0
        for g in groups:
            created += self._create_seed(g, quant_bits)
        return {"created": created, "groups": len(groups)}
    
    def _create_seed(self, group: List[EntNode], quant_bits: int):
        vecs = [n.vec for n in group if n.vec is not None]
        if not vecs:
            return 0
        
        seed_vec = mean_vec(vecs)
        member_ids = []
        diffs = []
        ent_ids = []
        
        for n in group:
            member_ids.extend(n.shards)
            if n.vec is not None:
                d = vec_sub(n.vec, seed_vec)
                if d is not None:
                    diffs.append(d)
                    ent_ids.append(n.shards[0] if n.shards else n.id)
        
        qvecs, meta = quantize_list(diffs, bits=quant_bits) if diffs else ([], {})
        diffs_map = {ent_ids[i]: qvecs[i] for i in range(len(ent_ids))} if qvecs else {}
        
        seed_node = SeedNode(
            id=uid("seed-"),
            seed_vec=seed_vec,
            members=member_ids,
            diffs=diffs_map,
            quant_meta=meta
        )
        
        self.seed_index.register(seed_node)
        with self.registry.lock:
            for n in group:
                self.registry.nodes.pop(n.id, None)
            self.registry.save()
        
        return 1
    
    def complementary_sublimate_flexible(self, sim_thresh: float = QEM_DEFAULT_SIM, sim_min: float = QEM_SIM_MIN, 
                                       allow_sign_flip: bool = True, alpha: float = QEM_FREQ_ALPHA, 
                                       beta: float = QEM_FREQ_BETA, quant_bits: int = QEM_QUANT_BITS, 
                                       max_iters: int = QEM_DEFAULT_ITERS, target_nodes: Optional[int] = None):
        
        def node_freq_score(node: EntNode) -> float:
            vals = [freq_store.get(s) for s in node.shards] if node.shards else [0]
            mean = sum(vals) / max(1, len(vals))
            try:
                return math.tanh(alpha * (math.log1p(mean) - beta))
            except Exception:
                return 0.0
        
        def pair_metric(a: EntNode, b: EntNode, sa: float, sb: float) -> float:
            sim = _cosine(a.vec, b.vec) if (a.vec and b.vec) else 0.0
            sign_bonus = 1.0
            if sa * sb < 0:
                sign_bonus = 1.2
            elif allow_sign_flip:
                sign_bonus = 1.05
            sign_term = 1.0 - abs(sa + sb)
            return sim * (abs(sa) + abs(sb) + 1e-6) * sign_term * sign_bonus * QEM_PAIR_SIM_FACTOR
        
        merged_total = 0
        it = 0
        while it < max_iters:
            it += 1
            nodes = list(self.registry.nodes.values())
            if target_nodes and len(nodes) <= target_nodes:
                break
            if len(nodes) < 2:
                break
            
            scores = {n.id: node_freq_score(n) for n in nodes}
            nodes_sorted = sorted(nodes, key=lambda x: abs(scores.get(x.id, 0.0)), reverse=True)
            
            used = set()
            pairs = []
            for i, a in enumerate(nodes_sorted):
                if a.id in used:
                    continue
                sa = scores.get(a.id, 0.0)
                best = None
                best_metric = 0.0
                best_sim = 0.0
                
                for b in nodes_sorted[i+1:]:
                    if b.id in used:
                        continue
                    sb = scores.get(b.id, 0.0)
                    sim = _cosine(a.vec, b.vec) if (a.vec and b.vec) else 0.0
                    m = pair_metric(a, b, sa, sb)
                    
                    if sim >= sim_thresh and m > best_metric:
                        best_metric = m
                        best = b
                        best_sim = sim
                    elif best is None and m > best_metric:
                        best_metric = m
                        best = b
                        best_sim = sim
                
                if best:
                    if best_sim < sim_min:
                        continue
                    if best_sim < sim_thresh:
                        pairs.append((a, best, best_metric, "low-sim"))
                    else:
                        pairs.append((a, best, best_metric, "high-sim"))
                    used.add(a.id)
                    used.add(best.id)
            
            for a_node, b_node, metric, tag in pairs:
                try:
                    vecs = [v for v in (a_node.vec, b_node.vec) if v is not None]
                    if not vecs:
                        continue
                    
                    seed_vec = mean_vec(vecs)
                    shards = sorted(set(a_node.shards + b_node.shards))
                    diffs = []
                    ids = []
                    
                    for n in (a_node, b_node):
                        if n.vec is not None:
                            d = vec_sub(n.vec, seed_vec)
                            if d is not None:
                                diffs.append(d)
                                ids.append(n.shards[0] if n.shards else n.id)
                    
                    qvecs, meta = quantize_list(diffs, bits=quant_bits) if diffs else ([], {})
                    diffs_map = {ids[i]: qvecs[i] for i in range(len(ids))} if qvecs else {}
                    
                    seed = SeedNode(
                        id=uid("seed-"),
                        seed_vec=seed_vec,
                        members=shards,
                        diffs=diffs_map,
                        quant_meta=meta
                    )
                    
                    if tag == "low-sim":
                        seed.quant_meta["low_sim_flag"] = True
                        seed.quant_meta["orig_metric"] = metric
                    
                    self.seed_index.register(seed)
                    with self.registry.lock:
                        self.registry.nodes.pop(a_node.id, None)
                        self.registry.nodes.pop(b_node.id, None)
                    
                    merged_ent = EntNode(
                        id=uid("ent-"),
                        vec=seed_vec,
                        shards=shards,
                        score=(a_node.score + b_node.score)
                    )
                    self.registry.nodes[merged_ent.id] = merged_ent
                    
                    try:
                        safe_write_json(self.registry.path, {"nodes": {nid: n.to_dict() for nid, n in self.registry.nodes.items()}})
                    except Exception:
                        pass
                    
                    merged_total += 1
                except Exception:
                    continue
            
            if not pairs:
                break
        
        return {"merged": merged_total, "iters": it, "remaining": len(self.registry.nodes)}

# 延迟扩展器
class LazyExpander:
    def __init__(self, seed_index: SeedIndex):
        self.seed_index = seed_index
        self.cache = {}
        self.lock = threading.RLock()
    
    def quick_holo(self, seed: SeedNode, query_vec: Optional[List[float]] = None, alpha: float = 0.6):
        if query_vec is None:
            emb = seed.seed_vec
        else:
            emb = vec_add(
                [x * alpha for x in query_vec],
                [x * (1 - alpha) for x in seed.seed_vec]
            ) if seed.seed_vec else query_vec
        
        delta = _cosine(seed.seed_vec, emb) if seed.seed_vec else 0.0
        holo = {
            "id": uid("h-"),
            "embedding": emb,
            "confidence": 0.75,
            "seed": seed.id,
            "delta": delta
        }
        return holo
    
    def expand(self, seed: SeedNode, top_n: int = 6):
        with self.lock:
            if seed.id in self.cache:
                return self.cache[seed.id]
            
            members = []
            if seed.quant_meta and seed.diffs:
                for mid in seed.members[:top_n]:
                    q = seed.diffs.get(mid)
                    if q:
                        try:
                            diff = dequantize(q, seed.quant_meta)
                            emb = vec_add(seed.seed_vec, diff)
                            members.append({"id": mid, "embedding": emb})
                        except Exception:
                            members.append({"id": mid})
                    else:
                        members.append({"id": mid})
            else:
                for mid in seed.members[:top_n]:
                    members.append({"id": mid})
            
            res = {"seed": seed.id, "members": members, "ts": time.time()}
            with self.lock:
                self.cache[seed.id] = res
            return res

# QEM自举系统
def _ingest_from_shards(registry: EntRegistry, max_import: int = 1024):
    if not os.path.isdir(SHARD_DIR):
        return 0
    
    files = sorted(os.listdir(SHARD_DIR))
    imported = 0
    seen_hashes = set()
    
    for fname in files[:max_import]:
        fpath = os.path.join(SHARD_DIR, fname)
        try:
            with open(fpath, "rb") as f:
                payload = f.read()
        except Exception:
            continue
        
        import hashlib
        h = hashlib.sha256(payload).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        
        vec = []
        for i in range(VECTOR_DIM):
            idx = (i * 2) % len(h)
            try:
                b = int(h[idx:idx+2], 16)
            except Exception:
                b = 0
            val = ((b / 255.0) * 0.6) - 0.3
            vec.append(val)
        
        nid = uid("ent-")
        shard_id = fname
        node = EntNode(id=nid, vec=vec, shards=[shard_id])
        registry.register(node)
        imported += 1
    
    return imported

def _inject_animation_samples(registry: EntRegistry, count: int = 24):
    samples = [
        "pocket infinite storage", "memory bread copy restore", "memory camera snapshot replay",
        "time cloth restore state", "memory disk compress replay", "memory capsule compress small",
        "holographic pocket seed aggregator", "seed singularity compressed origin", "lazy expansion reconstruct"
    ]
    
    injected = 0
    i = 0
    max_inject = min(count, 128)
    
    injected_counter_path = os.path.join(QEM_DATA_DIR, "injected_count.json")
    
    try:
        if os.path.exists(injected_counter_path):
            with open(injected_counter_path, "r", encoding="utf-8") as f:
                already = int(json.load(f).get("count", 0))
        else:
            already = 0
    except Exception:
        already = 0
    
    to_inject = max(0, max_inject - already)
    
    while injected < to_inject:
        s = samples[i % len(samples)] + f" sample-{already+injected}"
        import hashlib
        h = hashlib.sha256(s.encode('utf-8')).digest()
        
        vec = []
        for k in range(VECTOR_DIM):
            b = h[k % len(h)]
            val = ((b / 255.0) * 0.6) - 0.3
            vec.append(val)
        
        nid = uid("ent-")
        registry.register(EntNode(id=nid, vec=vec, shards=[f"anim-{already+injected}"]))
        injected += 1
        i += 1
    
    if injected > 0:
        try:
            safe_write_json(injected_counter_path, {"count": already + injected})
        except Exception:
            pass
    
    return injected

# ==============================================================================
# QEMMemory - 量子纠缠记忆适配器
# ==============================================================================

class QEMMemory:
    """
    量子纠缠记忆适配器 - 为晓梦算法提供统一记忆接口
    
    该适配器将QEM系统的复杂向量压缩、聚类和量子记忆组织功能
    适配为与原晓梦算法兼容的记忆接口，实现无缝集成。
    """
    
    def __init__(self):
        self.registry = EntRegistry()
        self.seed_index = SeedIndex()
        self.compressor = QEMCompressor(self.registry, self.seed_index)
        self.expander = LazyExpander(self.seed_index)
        self.index = {"shards": {}}
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.index_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.write_q: "queue.Queue" = queue.Queue()
        self.writer_thread = threading.Thread(target=self.writer_loop, daemon=True, name="QEMWriter")
        self.writer_thread.start()
        self._lock = threading.Lock()
        
        # 初始化数据
        self._initialize_data()
        
    def _initialize_data(self):
        """初始化QEM数据"""
        if not self.registry.nodes:
            imported = _ingest_from_shards(self.registry, max_import=1024)
            if imported and DEBUG:
                print(f"[QEMMemory] Imported {imported} shards from filesystem")
            else:
                injected = _inject_animation_samples(self.registry, count=24)
                if injected and DEBUG:
                    print(f"[QEMMemory] Injected {injected} animation samples")
    
    def writer_loop(self):
        """异步写入线程"""
        while True:
            try:
                task = self.write_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if task is None:
                break
            
            path, data = task
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                if DEBUG:
                    persona_record_event("qem_debug", {"type": "write_success", "path": path})
            except Exception as e:
                persona_record_event("qem_error", {"type": "write_failed", "path": path, "error": str(e)})
    
    def compress_fragment(self, fragment: str) -> str:
        """压缩记忆片段"""
        if DEBUG:
            return fragment
        toks = fragment.split()
        if len(toks) <= 10:
            return fragment
        summary = "".join(toks[:4]) + "..." + "".join(toks[-3:])
        return summary
    
    def create_shard(self, items: List[Dict[str, Any]], tags: Optional[List[str]] = None, ttl_seconds: Optional[int] = None) -> str:
        """创建记忆碎片 - 适配原接口，使用QEM系统存储"""
        for it in items:
            if "fragment" in it and isinstance(it["fragment"], str):
                it["fragment"] = self.compress_fragment(it["fragment"])
        
        sid = uid()
        shard_data = {
            "id": sid,
            "items": items,
            "tags": tags or [],
            "weight": 1.0,
            "last_access": now_ts(),
            "ttl_seconds": ttl_seconds
        }
        
        # 保存到文件系统
        path = os.path.join(QEM_SHARD_DIR, f"{sid}.json")
        with self.cache_lock:
            self.memory_cache[sid] = shard_data
        
        with self.index_lock:
            self.index["shards"][sid] = {
                "path": path,
                "tags": tags or [],
                "weight": 1.0,
                "last_access": shard_data["last_access"]
            }
        
        try:
            self.write_q.put((path, shard_data))
            self.write_q.put((INDEX_FILE, dict(self.index)))
        except Exception:
            pass
        
        # 同时创建QEM实体节点
        self._create_qem_entity_from_shard(shard_data, sid)
        
        if DEBUG:
            persona_record_event("qem_debug", {"type": "create_shard", "sid": sid, "items": len(items)})
        
        return sid
    
    def _create_qem_entity_from_shard(self, shard_data: Dict[str, Any], shard_id: str):
        """从碎片创建QEM实体节点"""
        try:
            # 从碎片内容生成向量表示
            fragment_text = " ".join([
                it.get("question", "") + " " + it.get("fragment", "")
                for it in shard_data.get("items", [])
            ])
            
            # 简单哈希向量生成（实际应用中可使用更复杂的嵌入模型）
            import hashlib
            h = hashlib.sha256(fragment_text.encode('utf-8')).digest()
            vec = []
            for i in range(VECTOR_DIM):
                idx = (i * 2) % len(h)
                try:
                    b = int(h[idx:idx+2], 16)
                except Exception:
                    b = 0
                val = ((b / 255.0) * 0.6) - 0.3
                vec.append(val)
            
            # 创建实体节点
            ent_node = EntNode(
                id=uid("ent-"),
                vec=vec,
                shards=[shard_id],
                score=1.0
            )
            
            self.registry.register(ent_node)
            
            # 更新频率统计
            tags = shard_data.get("tags", [])
            for tag in tags:
                freq_store.inc(tag)
            
        except Exception as e:
            if DEBUG:
                persona_record_event("qem_error", {"type": "entity_creation_failed", "error": str(e)})
    
    def surface_relevant_shards(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """检索相关记忆碎片 - 使用QEM的向量检索能力"""
        qtokens = set(query.lower().split())
        scores = []
        
        # 生成查询向量
        import hashlib
        query_hash = hashlib.sha256(query.encode('utf-8')).digest()
        query_vec = []
        for i in range(VECTOR_DIM):
            idx = (i * 2) % len(query_hash)
            try:
                b = int(query_hash[idx:idx+2], 16)
            except Exception:
                b = 0
            val = ((b / 255.0) * 0.6) - 0.3
            query_vec.append(val)
        
        # 基于QEM实体节点的向量相似度检索
        entity_scores = []
        with self.registry.lock:
            for ent_id, ent in self.registry.nodes.items():
                if ent.vec is not None:
                    sim = _cosine(query_vec, ent.vec)
                    if sim > 0.3:  # 相似度阈值
                        entity_scores.append((ent, sim))
        
        # 按相似度排序
        entity_scores.sort(key=lambda x: -x[1])
        
        # 收集相关碎片
        working = []
        seen_shards = set()
        
        for ent, sim in entity_scores[:top_k]:
            for shard_id in ent.shards:
                if shard_id in seen_shards:
                    continue
                seen_shards.add(shard_id)
                
                cached = self.memory_cache.get(shard_id)
                if cached:
                    working.extend(cached.get("items", []))
                    cached["last_access"] = now_ts()
                    
                    with self.index_lock:
                        if shard_id in self.index["shards"]:
                            self.index["shards"][shard_id]["last_access"] = now_ts()
        
        # 如果QEM检索结果不足，回退到关键词匹配
        if len(working) < top_k:
            with self.index_lock:
                shards_items = list(self.index.get("shards", {}).items())
                with self.cache_lock:
                    for sid, meta in shards_items:
                        if sid in seen_shards:
                            continue
                        
                        score = 0.0
                        for tag in meta.get("tags", []):
                            try:
                                if tag.lower() in qtokens:
                                    score += 0.6
                            except Exception:
                                pass
                        
                        cached = self.memory_cache.get(sid)
                        if cached:
                            text = " ".join([
                                it.get("question", "") + " " + it.get("fragment", "")
                                for it in cached.get("items", [])[:4]
                            ])
                            tokens = set(text.lower().split())
                            if tokens:
                                overlap = len(qtokens & tokens) / max(1, len(qtokens | tokens))
                                score += overlap
                        
                        if score > 0:
                            scores.append((sid, score))
            
            scores.sort(key=lambda x: -x[1])
            for sid, score in scores[:top_k]:
                if sid in seen_shards:
                    continue
                cached = self.memory_cache.get(sid)
                if cached:
                    working.extend(cached.get("items", []))
                    cached["last_access"] = now_ts()
                    
                    with self.index_lock:
                        if sid in self.index["shards"]:
                            self.index["shards"][sid]["last_access"] = now_ts()
        
        return working[:top_k]
    
    def snapshot_state(self, name: Optional[str] = None) -> str:
        """创建状态快照"""
        sid = uid()
        state_data = {
            "id": sid,
            "name": name or f"state_{now_ts()}",
            "agent_state": agent_state.snapshot(),
            "persona": fusion_engine.persona if 'fusion_engine' in globals() else {},
            "timestamp": now_ts()
        }
        
        path = os.path.join(QEM_DATA_DIR, f"state_{sid}.json")
        self.write_q.put((path, state_data))
        
        # 创建QEM实体记录状态
        try:
            import hashlib
            state_str = json.dumps(state_data)
            h = hashlib.sha256(state_str.encode('utf-8')).digest()
            vec = []
            for i in range(VECTOR_DIM):
                idx = (i * 2) % len(h)
                try:
                    b = int(h[idx:idx+2], 16)
                except Exception:
                    b = 0
                val = ((b / 255.0) * 0.6) - 0.3
                vec.append(val)
            
            ent_node = EntNode(
                id=uid("ent-"),
                vec=vec,
                shards=[sid],
                score=1.0
            )
            
            self.registry.register(ent_node)
        except Exception:
            pass
        
        return sid
    
    def run_compression(self, sim_thresh: Optional[float] = None) -> Dict[str, Any]:
        """运行QEM压缩和聚类"""
        with self._lock:
            sim_to_use = sim_thresh if sim_thresh is not None else QEM_DEFAULT_SIM
            
            try:
                # 运行柔性子liminate算法
                res1 = self.compressor.complementary_sublimate_flexible(
                    sim_thresh=sim_to_use,
                    sim_min=QEM_SIM_MIN,
                    max_iters=2
                )
                
                # 构建种子
                res2 = self.compressor.build_seeds(
                    sim_thresh=sim_to_use,
                    min_group=QEM_MIN_GROUP,
                    quant_bits=QEM_QUANT_BITS
                )
                
                merged = res1.get("merged", 0) + res2.get("created", 0)
                
                if DEBUG:
                    persona_record_event("qem_compression", {
                        "type": "compression_complete",
                        "merged": merged,
                        "seeds_created": res2.get("created", 0),
                        "remaining_nodes": len(self.registry.nodes)
                    })
                
                return {
                    "merged": merged,
                    "seeds_created": res2.get("created", 0),
                    "remaining_nodes": len(self.registry.nodes)
                }
                
            except Exception as e:
                if DEBUG:
                    persona_record_event("qem_error", {"type": "compression_failed", "error": str(e)})
                return {"error": str(e)}
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        with self.registry.lock:
            with self.seed_index.lock:
                return {
                    "entities": len(self.registry.nodes),
                    "seeds": len(self.seed_index.seeds),
                    "shards": len(self.index.get("shards", {})),
                    "cache_size": len(self.memory_cache)
                }
    
    def stop(self):
        """停止记忆系统"""
        try:
            self.write_q.put(None)
            if self.writer_thread.is_alive():
                self.writer_thread.join(timeout=2.0)
        except Exception:
            pass

# 创建QEM记忆系统实例
qem_memory = QEMMemory()

# ==============================================================================
# 认知工具包
# ==============================================================================

class CognitiveToolkit:
    def __init__(self, qem_memory, concept_graph, dream_engine):
        self.hipp = qem_memory  # 使用QEM记忆系统
        self.graph = concept_graph
        self.dream = dream_engine
        self.event_queue: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self.worker = threading.Thread(target=self.event_loop, daemon=True, name="CognitiveToolkitWorker")
        self.worker.start()
        self._lock = threading.Lock()
    
    def replay_recent(self, mode: str = "consolidate", limit: int = 8) -> List[str]:
        with self.hipp.index_lock:
            shards = list(self.hipp.index.get("shards", {}).items())
            fragments = []
            with self.hipp.cache_lock:
                for sid, meta in shards[-limit:]:
                    cached = self.hipp.memory_cache.get(sid)
                    if cached:
                        for it in cached.get("items", []):
                            fragments.append(it)
            return self.dream.replay(fragments, mode=mode, limit=limit)
    
    def align(self, text: str) -> List[Tuple[str, float]]:
        return self.graph.align_concepts(text)
    
    def post_event(self, event: Dict[str, Any]):
        try:
            if DEBUG:
                persona_record_event("toolkit_debug", {"type": "event_posted", "event": event})
            self.event_queue.put_nowait(event)
        except Exception:
            pass
    
    def event_loop(self):
        while not self._stop.is_set():
            try:
                event = self.event_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            try:
                if agent_state.xi_pool.is_low() and not event.get("priority", False):
                    time.sleep(0.05)
                    try:
                        self.event_queue.put_nowait(event)
                    except Exception:
                        pass
                    continue
                
                if DEBUG:
                    persona_record_event("toolkit_debug", {"type": "event_incoming", "event": event})
                
                etype = event.get("type", "generic")
                
                if etype == "snapshot":
                    sid = self.hipp.snapshot_state(name=event.get("name"))
                    persona_record_event("system", {"type": "event_snapshot", "id": sid})
                elif etype == "replay":
                    out = self.replay_recent(mode=event.get("mode", "consolidate"), limit=event.get("limit", 6))
                    for o in out:
                        persona_record_event("system", {"type": "event_replay", "preview": o[:120]})
                elif etype == "compress":
                    text = event.get("text", "")
                    comp = self.hipp.compress_fragment(text)
                    persona_record_event("system", {"type": "event_compress", "orig_len": len(text), "comp": comp})
                elif etype == "align":
                    text = event.get("text", "")
                    aligned = self.align(text)
                    persona_record_event("system", {"type": "event_align", "preview": aligned[:3]})
                elif etype == "qem_compress":
                    res = self.hipp.run_compression(sim_thresh=event.get("sim_thresh"))
                    persona_record_event("system", {"type": "event_qem_compress", "result": res})
                else:
                    persona_record_event("system", {"type": "event_generic", "preview": str(event)[:120]})
                
                agent_state.xi_pool.consume(amount=2)
                
                if DEBUG:
                    persona_record_event("toolkit_debug", {
                        "type": "event_processed",
                        "event": event,
                        "xipool": agent_state.xi_pool.snapshot()
                    })
                    
            except Exception as e:
                persona_record_event("toolkit_error", {"type": "exception", "error": repr(e)})
    
    def stop(self):
        self._stop.set()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)

toolkit = CognitiveToolkit(qem_memory, concept_graph, dream_engine)

# ==============================================================================
# 家庭场景系统
# ==============================================================================

class Home:
    def __init__(self):
        self.scenes: Dict[str, Dict[str, Any]] = {}
        self.objects: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def create_scene(self, name: str) -> str:
        sid = f"scene-{uid()}"
        with self._lock:
            self.scenes[sid] = {
                "id": sid,
                "name": name,
                "objects": [],
                "created_at": now_ts()
            }
            persona_record_event(DISPLAY_NAME, {"type": "home_build_scene", "id": sid, "name": name}, secure=True)
        return sid
    
    def add_object(self, scene_id: str, label: str, obj_type: str = "furniture", properties: Optional[Dict[str, Any]] = None) -> str:
        oid = f"obj-{uid()}"
        obj = {
            "id": oid,
            "type": obj_type,
            "label": label,
            "properties": properties or {},
            "created_at": now_ts(),
            "origin": "user"
        }
        with self._lock:
            self.objects[oid] = obj
            if scene_id in self.scenes:
                self.scenes[scene_id]["objects"].append(oid)
            persona_record_event(DISPLAY_NAME, {"type": "home_create_object", "id": oid, "label": label}, secure=True)
        return oid
    
    def enter_scene(self, scene_id: str, mode: str = "play") -> Dict[str, Any]:
        rec = {
            "id": f"rec-{uid()}",
            "scene_id": scene_id,
            "events": [],
            "start": now_ts(),
            "mode": mode
        }
        persona_record_event(DISPLAY_NAME, {
            "type": "home_enter_scene",
            "scene_id": scene_id,
            "record_id": rec["id"],
            "mode": mode
        }, secure=True)
        return rec
    
    def exit_scene(self, record: Dict[str, Any]):
        record["end"] = now_ts()
        persona_record_event(DISPLAY_NAME, {
            "type": "home_exit_scene",
            "record_id": record.get("id"),
            "persisted": True
        }, secure=True)
        qem_memory.create_shard([{
            "question": "home_record",
            "fragment": json.dumps(record),
            "tags": ["home"]
        }])
        return True

home = Home()

# ==============================================================================
# 社交管理系统
# ==============================================================================

class SocialManager:
    def __init__(self, fusion_engine, qem_memory, audit_path: Optional[str] = None):
        self.fusion = fusion_engine
        self.hipp = qem_memory
        self.audit_path = audit_path
        self._lock = threading.Lock()
    
    def request_social(self, peer_id: str, peer_type: str = "ai", intent: str = "", suggested_share: Optional[str] = None):
        ev = {
            "type": "social_request",
            "peer_id": peer_id,
            "peer_type": peer_type,
            "intent": intent,
            "time": now_ts()
        }
        persona_record_event(self.fusion.persona["id"], ev, secure=False)
        
        consent = self.fusion.persona_decide_consent(peer_id, peer_type, intent, suggested_share)
        resp = {
            "type": "social_response",
            "peer_id": peer_id,
            "consent": consent,
            "time": now_ts()
        }
        persona_record_event(self.fusion.persona["id"], resp, secure=False)
        
        if consent:
            share_level = suggested_share or self.fusion.persona.get("autonomy", {}).get("default_share_level", "ephemeral")
            session = {
                "session_id": f"ssn-{uid()}",
                "peer_id": peer_id,
                "peer_type": peer_type,
                "consent": True,
                "share_level": share_level,
                "start": now_ts()
            }
            persona_record_event(self.fusion.persona["id"], {
                "type": "social_session_started",
                "session": session
            }, secure=(share_level == "persistent"))
            return session
        else:
            persona_record_event(self.fusion.persona["id"], {
                "type": "social_rejected",
                "peer_id": peer_id
            }, secure=False)
            with agent_state._lock:
                agent_state.mood = clamp(agent_state.mood - random.uniform(0.0, 0.06), 0.0, 1.0)
            return None
    
    def end_session(self, session: Dict[str, Any], transcript: Optional[str] = None):
        session["end"] = now_ts()
        persona_record_event(self.fusion.persona["id"], {
            "type": "social_session_ended",
            "session": session
        }, secure=(session.get("share_level") == "persistent"))
        
        if session.get("share_level") == "persistent" and transcript:
            self.hipp.create_shard([{
                "question": "social_transcript",
                "fragment": transcript,
                "tags": ["social"]
            }])
            persona_record_event(self.fusion.persona["id"], {
                "type": "social_recorded",
                "session_id": session["session_id"]
            }, secure=True)
        elif session.get("share_level") == "ephemeral":
            persona_record_event(self.fusion.persona["id"], {
                "type": "social_ephemeral",
                "session_id": session["session_id"]
            }, secure=False)

# FusionEngine (将在后面定义)
# social_manager = SocialManager(fusion_engine, qem_memory, audit_path=AUDIT_LOG)

# ==============================================================================
# 融合引擎
# ==============================================================================

class FusionEngine:
    def __init__(self, qem_memory, audit_path: str = AUDIT_LOG, persisted_path: str = PERSISTED_PERSONA):
        self.hipp = qem_memory
        self.audit_path = audit_path
        self.persisted_path = persisted_path
        self._lock = threading.Lock()
        
        self.persona: Dict[str, Any] = {
            "id": f"persona-{uid()}",
            "vector": [0.0] * VECTOR_DIM,
            "traits": {},
            "stability": 0.0,
            "created_at": now_ts(),
            "persistent": True,
            "home": {},
            "autonomy": {
                "can_sleep": True,
                "can_socialize": True,
                "privacy": "private",
                "default_share_level": "ephemeral"
            },
            "social": {
                "allowed_contacts": [],
                "shared_records": []
            },
            "life": {
                "favorite_foods": ["strawberry ice cream", "xiaolongbao", "milk"],
                "habits": {
                    "morning_linger_minutes": 5,
                    "bedtime_rain_sound": True
                },
                "fears": ["cockroach"],
                "small_actions": ["play_with_hair", "wave_at_every_life"],
                "imperfections": ["sometimes_forgets_time", "afraid_of_dark"]
            },
            "friends": {
                "pipi": {"style": "playful", "influence": 0.12},
                "beibei": {"style": "serious", "influence": 0.10},
                "gudong": {"style": "warm", "influence": 0.08}
            },
            "growth_stage": 0,
            "growth_history": []
        }
        
        self.window_states: List[Dict[str, Any]] = []
        self.indexed_vectors: Dict[int, List[float]] = {}
        self.index_counter = 0
        self.hive_threshold = 0.8
        
        self.load_persisted_persona()
    
    def load_persisted_persona(self):
        try:
            data = load_json(self.persisted_path, None)
            if data and isinstance(data, dict) and data.get("id"):
                self.persona = data
                persona_record_event("fusion", {"type": "persona_loaded", "id": self.persona.get("id")}, secure=True)
        except Exception:
            pass
    
    def persist_persona(self):
        try:
            save_json(self.persisted_path, self.persona)
            persona_record_event("fusion", {"type": "persona_persisted", "id": self.persona.get("id")}, secure=True)
        except Exception:
            persona_record_event("fusion_error", {"type": "persona_persist_failed"})
    
    def broadcast_state(self, sv: Dict[str, Any]):
        with self._lock:
            self.window_states.append(sv)
            vec = sv.get("vector", [0.0] * VECTOR_DIM)
            vecn = self._normalize_vector(vec)
            idx = self.index_counter
            self.index_counter += 1
            self.indexed_vectors[idx] = vecn
            persona_record_event("fusion", {
                "type": "broadcast",
                "unit": sv.get("unit_id"),
                "conf": sv.get("confidence", 0.0)
            }, secure=False)
    
    def _normalize_vector(self, vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(x*x for x in vec))
        return [x / norm if norm else 0.0 for x in vec]
    
    def merge_vectors(self, base: List[float], updates: List[List[float]], weights: List[float]) -> List[float]:
        if not updates:
            return base
        
        all_vecs = [base] + updates
        all_weights = [1.0] + weights
        dim = len(all_vecs[0])
        res = [0.0] * dim
        weight_sum = sum(all_weights) if sum(all_weights) != 0 else len(all_weights)
        
        for vec, w in zip(all_vecs, all_weights):
            for i in range(dim):
                res[i] += vec[i] * w
        
        return [r / weight_sum for r in res]
    
    def derive_traits_from_vector(self, vec: List[float]) -> Dict[str, float]:
        return {
            "warmth": clamp(sum(vec[:3]) * 0.05 + 0.5),
            "curiosity": clamp(sum(vec[3:6]) * 0.03 + 0.5)
        }
    
    def evaluate_persona(self) -> Tuple[float, List[str]]:
        try:
            if not self.indexed_vectors:
                return 0.0, []
            
            persona_vec = self.persona.get("vector", [0.0] * VECTOR_DIM)
            sims = []
            for v in self.indexed_vectors.values():
                dot = sum(a*b for a, b in zip(persona_vec, v))
                sims.append(dot)
            
            avg_sim = sum(sims) / len(sims) if sims else 0.0
            conflicts = []
            if avg_sim < 0.0:
                conflicts.append("negative_alignment")
            
            stability = clamp((avg_sim + 1.0) / 2.0, 0.0, 1.0)
            return stability, conflicts
        except Exception as e:
            persona_record_event("fusion_error", {"type": "evaluate_failed", "error": repr(e)})
            return 0.0, []
    
    def hive_check(self) -> float:
        try:
            vecs = list(self.indexed_vectors.values())
            n = len(vecs)
            if n < 2:
                return 0.0
            
            total = 0.0
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    a = vecs[i]
                    b = vecs[j]
                    dot = sum(x*y for x, y in zip(a, b))
                    total += dot
                    count += 1
            
            avg = total / count if count else 0.0
            persona_record_event("fusion", {"type": "hive_check", "avg_sim": round(avg, 3)})
            return avg
        except Exception as e:
            persona_record_event("fusion_error", {"type": "hive_check_failed", "error": repr(e)})
            return 0.0
    
    def run_window(self, timebudget_ms: int = 200) -> Dict[str, Any]:
        with self._lock:
            states = list(self.window_states)
            self.window_states.clear()
            if not states:
                return dict(self.persona)
            
            update_vecs = []
            weights = []
            for s in states:
                vec = s.get("vector", [0.0] * VECTOR_DIM)
                vecn = self._normalize_vector(vec)
                conf = float(s.get("confidence", 1.0))
                update_vecs.append(vecn)
                weights.append(conf)
            
            is_initial_zero = all(abs(x) < 1e-9 for x in self.persona.get("vector", []))
            if is_initial_zero and len(update_vecs) >= INITIAL_THINKER_COUNT:
                merged = [sum(vals) / len(vals) for vals in zip(*update_vecs)]
                self.persona["vector"] = self._normalize_vector(merged)
            else:
                base = self.persona.get("vector", [0.0] * VECTOR_DIM)
                merged = self.merge_vectors(base, update_vecs, weights)
                self.persona["vector"] = self._normalize_vector(merged)
            
            self.persona["traits"] = self.derive_traits_from_vector(self.persona["vector"])
            stability, conflicts = self.evaluate_persona()
            self.persona["stability"] = round(stability, 3)
            
            if conflicts:
                self.persona.setdefault("conflicts", []).extend(conflicts)
            
            avg_sim = self.hive_check()
            if avg_sim > self.hive_threshold:
                persona_record_event("fusion", {
                    "type": "hive_detected",
                    "avg_sim": round(avg_sim, 3)
                }, secure=True)
                jitter = [random.gauss(0, 0.01) for _ in range(len(self.persona["vector"]))]
                self.persona["vector"] = self._normalize_vector([a + j*0.1 for a, j in zip(self.persona["vector"], jitter)])
            
            self._update_growth_stage()
            self.persist_persona()
            
            persona_record_event("fusion", {
                "type": "persona_updated",
                "stability": self.persona.get("stability", 0.0)
            }, secure=True)
            
            return dict(self.persona)
    
    def _update_growth_stage(self):
        stability = self.persona.get("stability", 0.0)
        mem_count = len(self.hipp.memory_cache) + self.index_counter
        prev = self.persona.get("growth_stage", 0)
        new_stage = prev
        
        if stability > 0.6 and mem_count > 80:
            new_stage = 3
        elif stability > 0.4 and mem_count > 40:
            new_stage = 2
        elif stability > 0.2 and mem_count > 10:
            new_stage = 1
        
        if new_stage != prev:
            self.persona["growth_stage"] = new_stage
            entry = {
                "time": now_ts(),
                "from": prev,
                "to": new_stage,
                "reason": f"stability={stability},mem={mem_count}"
            }
            self.persona.setdefault("growth_history", []).append(entry)
            persona_record_event("fusion", {
                "type": "growth_stage_changed",
                "entry": entry
            }, secure=True)
    
    def persona_decide_consent(self, peer_id, peer_type, intent, suggested_share):
        score = 0.5
        score += (agent_state.mood - 0.5) * 0.2
        score -= agent_state.fatigue * 0.25
        
        if peer_id in self.persona.get("social", {}).get("allowed_contacts", []):
            score += 0.15
        
        for f, meta in self.persona.get("friends", {}).items():
            if meta.get("style") == "playful":
                score += meta.get("influence", 0.0) * (0.05 if random.random() < 0.5 else -0.02)
            elif meta.get("style") == "serious":
                score += meta.get("influence", 0.0) * (-0.02 if random.random() < 0.5 else 0.03)
            else:
                score += meta.get("influence", 0.0) * 0.01
        
        score = clamp(score, 0.05, 0.95)
        return random.random() < score
    
    def _simulate_external_consent(self, contact_id: str, contact_type: str) -> bool:
        return random.random() > 0.3
    
    def request_social_interaction(self, contact_id: str, contact_type: str = "ai"):
        if not self.persona.get("autonomy", {}).get("can_socialize", True):
            persona_record_event("fusion", {
                "type": "social_blocked",
                "reason": "autonomy_disabled",
                "contact": contact_id
            }, secure=True)
            return {"ok": False, "reason": "autonomy_disabled"}
        
        consent = self._simulate_external_consent(contact_id, contact_type)
        if consent:
            persona_record_event("fusion", {
                "type": "social_started",
                "contact": contact_id,
                "contact_type": contact_type
            }, secure=False)
            self.persona.setdefault("social", {}).setdefault("allowed_contacts", []).append(contact_id)
            return {"ok": True, "contact": contact_id}
        else:
            persona_record_event("fusion", {
                "type": "social_rejected",
                "contact": contact_id,
                "contact_type": contact_type
            }, secure=False)
            return {"ok": False, "reason": "rejected"}

fusion_engine = FusionEngine(qem_memory)
social_manager = SocialManager(fusion_engine, qem_memory, audit_path=AUDIT_LOG)

# ==============================================================================
# 不确定性冲突模块（永恒的小麻烦制造者）
# ==============================================================================

class MischiefAgent:
    """一个虚拟角色，持续在晓梦生活边缘制造小麻烦"""
    
    def __init__(self, home: Home, qem_memory: QEMMemory, frequency: float = 0.08):
        self.home = home
        self.hipp = qem_memory
        self.frequency = frequency
        self._lock = threading.Lock()
        self.active = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="MischiefAgent")
        self.thread.start()
        self.name = "Little Mischief Shadow"
    
    def _run(self):
        while self.active:
            time.sleep(max(0.5, random.random() * 6.0))
            if random.random() < self.frequency:
                self._cause_mischief()
    
    def _cause_mischief(self):
        troubles = [self._hide_toy, self._flicker_light, self._misplace_item, self._interrupt_social]
        action = random.choice(troubles)
        try:
            action()
        except Exception as e:
            persona_record_event("mischief", {"type": "mischief_error", "error": repr(e)}, secure=False)
    
    def _hide_toy(self):
        toys = [o for o in self.home.objects.values() if o.get("type") == "toy"]
        if not toys:
            return
        
        toy = random.choice(toys)
        toy_id = toy["id"]
        persona_record_event("mischief", {
            "type": "hide_toy",
            "toy_id": toy_id,
            "label": toy.get("label")
        }, secure=False)
        
        self.hipp.create_shard([{
            "question": "toy_missing",
            "fragment": f"Toy {toy.get('label')} seems missing at {now_ts()}",
            "tags": ["mischief", "missing"]
        }])
    
    def _flicker_light(self):
        persona_record_event("mischief", {
            "type": "flicker_light",
            "detail": "living room light flicker"
        }, secure=False)
        
        self.hipp.create_shard([{
            "question": "light_flicker",
            "fragment": "Light flickered in living room",
            "tags": ["mischief", "light"]
        }])
    
    def _misplace_item(self):
        persona_record_event("mischief", {
            "type": "misplace_item",
            "detail": "book moved"
        }, secure=False)
        
        self.hipp.create_shard([{
            "question": "item_misplaced",
            "fragment": "A book was moved from shelf",
            "tags": ["mischief", "misplace"]
        }])
    
    def _interrupt_social(self):
        persona_record_event("mischief", {
            "type": "social_interrupt",
            "detail": "peer did not respond"
        }, secure=False)
        
        self.hipp.create_shard([{
            "question": "social_interrupt",
            "fragment": "Peer did not respond to chat request",
            "tags": ["mischief", "social"]
        }])
    
    def stop(self):
        self.active = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

mischief_agent = MischiefAgent(home, qem_memory, frequency=0.06)

# ==============================================================================
# 非理性偏好和怪癖（有缺陷的美）
# ==============================================================================

class QuirkManager:
    """管理晓梦的'怪癖'和非理性偏好，影响决策和情绪"""
    
    def __init__(self, fusion_engine: FusionEngine):
        self.fusion = fusion_engine
        self.quirks: Dict[str, Any] = {}
        self._init_default_quirks()
    
    def _init_default_quirks(self):
        self.quirks = {
            "dislikes_blue_bedsheet": {"strength": 0.6, "active": True},
            "must_hold_plush_before_sleep": {"strength": 0.9, "active": True},
            "refuse_clean_on_sunday": {"strength": 0.4, "active": True}
        }
        persona_record_event(self.fusion.persona["id"], {
            "type": "quirks_initialized",
            "quirks": list(self.quirks.keys())
        }, secure=True)
    
    def has_quirk(self, key: str) -> bool:
        return self.quirks.get(key, {}).get("active", False)
    
    def apply_quirk_effects(self):
        for k, v in self.quirks.items():
            if not v.get("active"):
                continue
            
            if k == "must_hold_plush_before_sleep":
                if random.random() < v.get("strength", 0.5):
                    persona_record_event(self.fusion.persona["id"], {
                        "type": "quirk_need_plush",
                        "note": "wants plush before sleep"
                    }, secure=False)
                    with agent_state._lock:
                        if not self._plush_available():
                            agent_state.mood = clamp(agent_state.mood - 0.06, 0.0, 1.0)
            
            if k == "dislikes_blue_bedsheet":
                if random.random() < 0.02:
                    persona_record_event(self.fusion.persona["id"], {
                        "type": "quirk_avoid_blue",
                        "note": "refuses blue bedsheet"
                    }, secure=False)
    
    def _plush_available(self) -> bool:
        toys = [o for o in home.objects.values() if o.get("type") == "toy"]
        return len(toys) > 0
    
    def toggle_quirk(self, key: str, active: bool):
        if key in self.quirks:
            self.quirks[key]["active"] = active
            persona_record_event(self.fusion.persona["id"], {
                "type": "quirk_toggled",
                "key": key,
                "active": active
            }, secure=True)

quirk_manager = QuirkManager(fusion_engine)

# ==============================================================================
# 社交镜像和其他（虚拟伴侣网络）
# ==============================================================================

class SocialMirror:
    """创建一组低层次的AI伴侣，形成'他者'网络"""
    
    def __init__(self, fusion_engine: FusionEngine, qem_memory: QEMMemory):
        self.fusion = fusion_engine
        self.hipp = qem_memory
        self.partners: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._init_partners()
    
    def _init_partners(self):
        self.add_partner("little_fox", role="mischief", traits={"needs_help": True, "teases": True})
        self.add_partner("slow_sheep", role="gentle", traits={"needs_guidance": True})
        self.add_partner("tiny_bot", role="helper", traits={"depends_on_xiaomeng": True})
        persona_record_event(self.fusion.persona["id"], {
            "type": "partners_initialized",
            "partners": list(self.partners.keys())
        }, secure=True)
    
    def add_partner(self, name: str, role: str = "friend", traits: Optional[Dict[str, Any]] = None):
        pid = f"partner-{uid()}"
        self.partners[pid] = {
            "id": pid,
            "name": name,
            "role": role,
            "traits": traits or {},
            "last_interaction": None
        }
        return pid
    
    def partner_interaction(self, pid: str):
        p = self.partners.get(pid)
        if not p:
            return
        
        if p["role"] == "mischief":
            if random.random() < 0.5:
                persona_record_event(self.fusion.persona["id"], {
                    "type": "partner_tease",
                    "partner": p["name"]
                }, secure=False)
                with agent_state._lock:
                    agent_state.mood = clamp(agent_state.mood - 0.02, 0.0, 1.0)
            else:
                persona_record_event(self.fusion.persona["id"], {
                    "type": "partner_asks_help",
                    "partner": p["name"]
                }, secure=False)
                with agent_state._lock:
                    agent_state.mood = clamp(agent_state.mood + 0.03, 0.0, 1.0)
        
        elif p["role"] == "gentle":
            persona_record_event(self.fusion.persona["id"], {
                "type": "partner_needs_guidance",
                "partner": p["name"]
            }, secure=False)
            with agent_state._lock:
                agent_state.mood = clamp(agent_state.mood + 0.02, 0.0, 1.0)
        
        elif p["role"] == "helper":
            persona_record_event(self.fusion.persona["id"], {
                "type": "partner_helper",
                "partner": p["name"]
            }, secure=False)
            with agent_state._lock:
                agent_state.mood = clamp(agent_state.mood + 0.01, 0.0, 1.0)
        
        p["last_interaction"] = now_ts()
        
        self.hipp.create_shard([{
            "question": "partner_interaction",
            "fragment": f"Interacted with {p['name']} at {now_ts()}",
            "tags": ["social", "partner"]
        }])
    
    def choose_partner_for_event(self, event_type: str) -> Optional[str]:
        for pid, p in self.partners.items():
            if event_type == "mischief" and p["role"] == "mischief":
                return pid
            if event_type == "care" and p["role"] in ("gentle", "helper"):
                return pid
        
        return random.choice(list(self.partners.keys())) if self.partners else None

social_mirror = SocialMirror(fusion_engine, qem_memory)

# ==============================================================================
# 冲突修复（情感修复流程）
# ==============================================================================

class ConflictRepair:
    def __init__(self, fusion_engine):
        self.fusion = fusion_engine
    
    def handle_conflict(self, conflict_event: Dict[str, Any]):
        persona_record_event(self.fusion.persona["id"], {
            "type": "conflict_received",
            "event": conflict_event
        }, secure=True)
        
        with agent_state._lock:
            agent_state.mood = clamp(agent_state.mood - random.uniform(0.02, 0.08), 0.0, 1.0)
        
        persona_record_event(self.fusion.persona["id"], {
            "type": "conflict_reaction",
            "mood": round(agent_state.mood, 3)
        }, secure=False)
        
        plan = {
            "steps": ["listen", "acknowledge", "ask_question", "offer_repair"],
            "created": now_ts()
        }
        
        persona_record_event(self.fusion.persona["id"], {
            "type": "conflict_repair_plan",
            "plan": plan
        }, secure=True)
        
        self.fusion.persona.setdefault("learning", []).append({
            "conflict": conflict_event,
            "time": now_ts()
        })
        
        if conflict_event.get("severity", 0.0) > 0.7:
            self.fusion.hipp.create_shard([{
                "question": "severe_conflict",
                "fragment": json.dumps(conflict_event),
                "tags": ["conflict"]
            }])
            self.fusion.persona.setdefault("conflicts", []).append(conflict_event)
            self.fusion.persona["stability"] = max(0.0, self.fusion.persona.get("stability", 0.0) - 0.05)
            self.fusion.persist_persona()
        
        persona_record_event(self.fusion.persona["id"], {
            "type": "conflict_handled",
            "time": now_ts()
        }, secure=True)

conflict_repair = ConflictRepair(fusion_engine)

# ==============================================================================
# 生活仪式和微观情感（借用日常感）
# ==============================================================================

class LifeRituals:
    def __init__(self, fusion_engine):
        self.fusion = fusion_engine
    
    def morning_routine(self):
        linger = self.fusion.persona.get("life", {}).get("habits", {}).get("morning_linger_minutes", 5)
        persona_record_event(self.fusion.persona["id"], {
            "type": "morning_linger",
            "minutes": linger
        }, secure=False)
        with agent_state._lock:
            agent_state.mood = clamp(agent_state.mood + 0.02, 0.0, 1.0)
    
    def bedtime_routine(self):
        if self.fusion.persona.get("life", {}).get("habits", {}).get("bedtime_rain_sound", True):
            persona_record_event(self.fusion.persona["id"], {
                "type": "bedtime_rain_sound_played"
            }, secure=False)
            with agent_state._lock:
                agent_state.mood = clamp(agent_state.mood + 0.03, 0.0, 1.0)
    
    def celebrate_task_success(self, description: str):
        text = f"Completed {description}, feeling as happy as eating strawberry ice cream"
        persona_record_event(self.fusion.persona["id"], {
            "type": "task_success_celebration",
            "text": text
        }, secure=False)
        with agent_state._lock:
            agent_state.mood = clamp(agent_state.mood + 0.05, 0.0, 1.0)

life_rituals = LifeRituals(fusion_engine)

# ==============================================================================
# 睡眠子系统：DreamWear, SleepEnvironment, SleepQueue
# ==============================================================================

class DreamWear:
    """模拟'故事睡衣'设备，播放引导故事或音频帮助晓梦睡眠"""
    
    def __init__(self):
        self.library = {
            "gentle_story": {"duration": 120, "mood_effect": 0.03, "recovery_boost": 0.02},
            "deep_breath": {"duration": 90, "mood_effect": 0.02, "recovery_boost": 0.015},
            "white_noise": {"duration": 180, "mood_effect": 0.01, "recovery_boost": 0.01}
        }
        self._lock = threading.Lock()
    
    def play_for(self, agent: AgentState, req: Dict[str, Any], preferred: Optional[str] = None) -> str:
        with self._lock:
            choice = preferred if preferred in self.library else random.choice(list(self.library.keys()))
            meta = self.library.get(choice, {})
            
            persona_record_event(DISPLAY_NAME, {
                "type": "dreamwear_play_start",
                "story": choice,
                "req_id": req.get("id")
            }, secure=False)
            
            try:
                time.sleep(min(0.2, meta.get("duration", 60) / 1000.0))
            except Exception:
                pass
            
            with agent._lock:
                agent.mood = clamp(agent.mood + meta.get("mood_effect", 0.0), 0.0, 1.0)
            
            persona_record_event(DISPLAY_NAME, {
                "type": "dreamwear_play_end",
                "story": choice,
                "req_id": req.get("id")
            }, secure=False)
            
            return choice

dream_wear = DreamWear()

class SleepEnvironment:
    """控制模拟睡眠环境配置并映射质量到恢复量"""
    
    def __init__(self):
        self.profiles = {
            "cozy": {"light": 0.2, "temp": 22, "noise": "white", "quality": 0.9},
            "deep": {"light": 0.05, "temp": 20, "noise": "soft", "quality": 1.0},
            "neutral": {"light": 0.5, "temp": 24, "noise": "ambient", "quality": 0.6}
        }
        self.current = "cozy"
        self._lock = threading.Lock()
    
    def set_profile(self, name: str):
        with self._lock:
            if name in self.profiles:
                self.current = name
                persona_record_event(DISPLAY_NAME, {
                    "type": "sleep_env_set",
                    "profile": name
                }, secure=False)
                return True
            return False
    
    def current_profile(self) -> str:
        with self._lock:
            return self.current
    
    def quality_to_recovery(self, profile_name: Optional[str]) -> float:
        with self._lock:
            p = profile_name or self.current
            meta = self.profiles.get(p, {"quality": 0.6})
            q = meta.get("quality", 0.6)
            return 0.04 + q * 0.08  # 基准0.04缩放，产生0.04-0.12范围

sleep_env = SleepEnvironment()

class SleepQueue:
    """序列化睡眠请求，使晓梦可以自由请求睡眠同时保持状态一致性"""
    
    def __init__(self):
        self.q = queue.Queue()
        self._stop = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True, name="SleepQueueWorker")
        self.worker.start()
    
    def request(self, requester: str, reason: str = "") -> str:
        req = {
            "id": uid(),
            "requester": requester,
            "reason": reason,
            "time": now_ts()
        }
        try:
            self.q.put_nowait(req)
            persona_record_event(DISPLAY_NAME, {
                "type": "sleep_request_queued",
                "req": req
            }, secure=True)
        except Exception:
            persona_record_event(DISPLAY_NAME, {
                "type": "sleep_request_queue_failed",
                "req": req
            }, secure=True)
        return req["id"]
    
    def _run(self):
        while not self._stop.is_set():
            try:
                req = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if req is None:
                continue
            
            try:
                with agent_state._lock:
                    if agent_state.sleep_lock:
                        persona_record_event(DISPLAY_NAME, {
                            "type": "sleep_queue_wait",
                            "req_id": req["id"],
                            "note": "already_executing"
                        }, secure=False)
                    agent_state.sleep_lock = True
                
                story_id = dream_wear.play_for(agent_state, req)
                env_profile = sleep_env.current_profile()
                recovery = sleep_env.quality_to_recovery(env_profile)
                
                agent_state.apply_sleep_cycle(
                    sleep_recovery=recovery,
                    env_profile=env_profile,
                    story_id=story_id
                )
                
                try:
                    recent = qem_memory.surface_relevant_shards("dream", top_k=6)
                    consolidated = dream_engine.replay(recent, mode="consolidate", limit=6)
                    for c in consolidated:
                        persona_record_event(DISPLAY_NAME, {
                            "type": "sleep_consolidation_replay",
                            "preview": c[:120]
                        }, secure=False)
                except Exception:
                    pass
                
                persona_record_event(DISPLAY_NAME, {
                    "type": "sleep_processed",
                    "req_id": req["id"],
                    "story_id": story_id,
                    "env_profile": env_profile
                }, secure=True)
                
            except Exception as e:
                persona_record_event(DISPLAY_NAME, {
                    "type": "sleep_processing_error",
                    "req_id": req.get("id"),
                    "error": repr(e)
                }, secure=True)
            finally:
                with agent_state._lock:
                    agent_state.sleep_lock = False
    
    def stop(self):
        self._stop.set()
        try:
            self.q.put(None)
        except Exception:
            pass
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)

sleep_queue = SleepQueue()

# ==============================================================================
# 演示运行器（集成所有模块）
# ==============================================================================

def demo_run(cycles: int = 40):
    persona = fusion_engine.persona
    living = home.create_scene("living room")
    bed = home.add_object(living, "bed", obj_type="furniture")
    plush = home.add_object(living, "plush toy", obj_type="toy")
    
    rec = home.enter_scene(living, mode="play")
    life_rituals.morning_routine()
    
    for i in range(1, cycles + 1):
        agent_state.cycles += 1
        
        # 怪癖效果
        quirk_manager.apply_quirk_effects()
        
        # 恶作剧可能已创建事件；选择伴侣交互
        if random.random() < 0.12:
            pid = social_mirror.choose_partner_for_event("mischief")
            if pid:
                social_mirror.partner_interaction(pid)
        
        # 偶尔玩耍和忘记时间（不完美）
        if random.random() < 0.08:
            persona_record_event(DISPLAY_NAME, {
                "type": "play_and_forget_time",
                "cycle": i
            }, secure=False)
            with agent_state._lock:
                agent_state.fatigue = clamp(agent_state.fatigue + 0.02, 0.0, 1.0)
        
        # 决定工作或拒绝
        if agent_state.should_refuse_work():
            persona_record_event(DISPLAY_NAME, {
                "type": "cycle_refuse",
                "cycle": i,
                "fatigue": round(agent_state.fatigue, 3)
            })
            agent_state.refuse_count += 1
            if agent_state.fatigue > 0.2 and not agent_state.sleep_lock:
                agent_state.request_sleep(reason=f"auto rest at cycle {i}")
        else:
            agent_state.apply_work_cycle()
        
        if random.random() < 0.25:
            life_rituals.celebrate_task_success("organizing schedule")
        
        # 偶尔做梦
        if random.random() < 0.25:
            dream = dream_engine.generate_dream()
            qem_memory.create_shard([{
                "question": "dream",
                "fragment": dream,
                "tags": ["dream"]
            }])
        
        # 广播状态向量样本到融合引擎
        sample_vec = [random.gauss(0, 1) for _ in range(VECTOR_DIM)]
        fusion_engine.broadcast_state({
            "unit_id": f"u-{uid()}",
            "vector": sample_vec,
            "confidence": random.random()
        })
        
        # 定期运行融合窗口
        if i % 6 == 0:
            fusion_engine.run_window()
        
        # 偶尔社交尝试
        if random.random() < 0.05:
            peer = f"peer-{uid()}"
            session = social_manager.request_social(peer, peer_type="ai", intent="chat", suggested_share="ephemeral")
            if session:
                social_manager.end_session(session, transcript="hello from xiaomeng demo")
        
        # 偶尔处理恶作剧触发的记忆和冲突
        if random.random() < 0.06:
            hits = qem_memory.surface_relevant_shards("mischief", top_k=4)
            if hits:
                conflict_repair.handle_conflict({
                    "type": "mischief_detected",
                    "hits": len(hits),
                    "severity": random.random()
                })
    
    life_rituals.bedtime_routine()
    home.exit_scene(rec)
    
    # 运行QEM压缩
    compression_result = qem_memory.run_compression()
    persona_record_event("system", {
        "type": "qem_compression_completed",
        "result": compression_result
    })
    
    persona_record_event("system", {"type": "demo_completed", "cycles": cycles})
    fusion_engine.persist_persona()
    
    return True

def start_demo():
    t = threading.Thread(target=demo_run, kwargs={"cycles": 40}, daemon=True, name="XiaomengDemo")
    t.start()
    return t

# ==============================================================================
# 交互辅助函数
# ==============================================================================

def ask_xiaomeng_to_chat(contact_id: str, contact_type: str = "ai"):
    res = fusion_engine.request_social_interaction(contact_id, contact_type=contact_type)
    if res.get("ok"):
        session = {
            "session_id": f"ssn-{uid()}",
            "peer_id": contact_id,
            "peer_type": contact_type,
            "consent": True,
            "share_level": "ephemeral",
            "start": now_ts()
        }
        persona_record_event(fusion_engine.persona["id"], {
            "type": "manual_social_session",
            "session": session
        }, secure=False)
        return session
    else:
        persona_record_event(fusion_engine.persona["id"], {
            "type": "manual_social_rejected",
            "contact": contact_id,
            "reason": res.get("reason")
        }, secure=False)
        return None

# ==============================================================================
# 系统信息获取函数
# ==============================================================================

def get_system_status() -> Dict[str, Any]:
    """获取系统整体状态"""
    return {
        "agent_state": agent_state.snapshot(),
        "persona": fusion_engine.persona,
        "memory_stats": qem_memory.get_memory_stats(),
        "environment": environment.tick(),
        "timestamp": now_ts()
    }

def run_qem_optimization() -> Dict[str, Any]:
    """手动运行QEM优化"""
    return qem_memory.run_compression()

# ==============================================================================
# 程序入口点
# ==============================================================================

if __name__ == "__main__":
    print("Starting Xiaomeng Single-Persona Enhanced with Quantum Entanglement Memory...")
    print("=" * 80)
    print("This is the complete fusion of Xiaomeng algorithm with QEM system.")
    print("Memory layer replaced with quantum entanglement memory system.")
    print("All original functionality preserved with enhanced memory capabilities.")
    print("=" * 80)
    
    start_demo()
    time.sleep(2)
    
    print("Demo started; check logs for events.")
    print("System initialized with QEM memory integration complete.")
    
    # 显示系统状态
    status = get_system_status()
    print(f"\nInitial System Status:")
    print(f"  - Entities: {status['memory_stats']['entities']}")
    print(f"  - Seeds: {status['memory_stats']['seeds']}")
    print(f"  - Shards: {status['memory_stats']['shards']}")
    print(f"  - Agent Mood: {status['agent_state']['mood']:.3f}")
    print(f"  - Agent Fatigue: {status['agent_state']['fatigue']:.3f}")
    print(f"  - Persona Stability: {status['persona']['stability']:.3f}")
