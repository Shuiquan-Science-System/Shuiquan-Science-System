# -*- coding: utf-8 -*-
""" Xiaomeng — Single‑Persona Soul Algorithm (Enhanced with Sleep Inventions)
Python 3.8+
This file merges the previously provided modules and integrates an enhanced sleep subsystem inspired
by "story sleepwear", sleep environment control, a sleep request queue, and memory consolidation hooks.
All original logic is preserved unless explicitly modified to integrate the new sleep flow.
"""

import os
import time
import random
import json
import uuid
import threading
import queue
import sqlite3
import math
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# -------------------------
# Basic configuration and paths
# -------------------------
DEBUG = True
DISPLAY_NAME = "Xiaomeng"
BASE_DIR = os.path.abspath(".")
SHARD_DIR = os.path.join(BASE_DIR, "shards")
INDEX_FILE = os.path.join(BASE_DIR, "hybrid_index.json")
DISTRIBUTED_DB = os.path.join(BASE_DIR, "hybrid_distributed.db")
PERSONA_FILE = os.path.join(BASE_DIR, "persona_ledger.json")
PERSISTED_PERSONA = os.path.join(BASE_DIR, "persona_single.json")
SECURE_LOG_DIR = os.path.join(BASE_DIR, "securelogs")
LOCAL_KEY_PATH = os.path.join(BASE_DIR, "localkey.bin")
AUDIT_LOG = os.path.join(BASE_DIR, "audit_log.jsonl")

VECTOR_DIM = 64
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

def ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

ensure_dir(SECURE_LOG_DIR)
ensure_dir(SHARD_DIR)

# -------------------------
# Utility functions
# -------------------------
def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def uid() -> str:
    return str(uuid.uuid4())[:8]

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

# -------------------------
# Keys and signatures
# -------------------------
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

# -------------------------
# Event recording (persona ledger + secure logs)
# -------------------------
def persona_record_event(entity_id: str, event: Dict[str, Any], secure: bool = False):
    try:
        ev = dict(event)
        if DEBUG:
            ev["debug_time"] = now_ts()
            ev["debug_thread"] = threading.current_thread().name
            try:
                ev["debug_preview"] = (json.dumps(ev, ensure_ascii=False)[:800] + "...") if len(json.dumps(ev, ensure_ascii=False)) > 800 else json.dumps(ev, ensure_ascii=False)
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

# -------------------------
# Core states: XiPool, TraumaManager, AgentState
# -------------------------
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
            return {"current": self.current, "max": self.max_pool, "ratio": round(self.current / self.max_pool if self.max_pool else 0.0, 3)}

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
        """
        Modified: instead of rejecting when sleep_lock is True, we enqueue the request into SleepQueue.
        Returns True if the request was accepted into the queue.
        """
        with self._lock:
            # If already executing, we still accept the request by queueing it.
            # The SleepQueue will process requests sequentially.
            req_id = sleep_queue.request(requester=DISPLAY_NAME, reason=reason)
            persona_record_event(DISPLAY_NAME, {"type": "sleep_request_accepted", "req_id": req_id, "reason": reason, "sleep_lock": self.sleep_lock, "fatigue": round(self.fatigue,3)}, secure=True)
            return True

    def apply_sleep_cycle(self, sleep_recovery: float = 0.08, env_profile: Optional[str] = None, story_id: Optional[str] = None):
        """
        Modified: apply_sleep_cycle now accepts a recovery amount (possibly computed from environment),
        triggers trauma decay, xi_pool recovery, memory consolidation, and releases sleep_lock at the end.
        """
        with self._lock:
            old = self.fatigue
            self.fatigue = max(0.0, self.fatigue - sleep_recovery)
            self.mood = min(1.0, self.mood + 0.02)
            self.xi_pool.recover()
            self.trauma_manager.decay()
            # Memory consolidation hook: create a shard summarizing recent sleep consolidation
            try:
                hippocampus.create_shard([{
                    "question": "sleep_consolidation",
                    "fragment": json.dumps({
                        "time": now_ts(),
                        "old_fatigue": round(old,3),
                        "new_fatigue": round(self.fatigue,3),
                        "env_profile": env_profile,
                        "story_id": story_id
                    }, ensure_ascii=False),
                    "tags": ["sleep","consolidation"]
                }])
            except Exception:
                pass
            # release sleep lock so future sleep requests are allowed
            self.sleep_lock = False
            persona_record_event(DISPLAY_NAME, {"type": "sleep_cycle", "old_fatigue": round(old,3), "new_fatigue": round(self.fatigue,3), "env_profile": env_profile, "story_id": story_id}, secure=True)

    def apply_work_cycle(self, produced: int = 1, fatigue_increase: float = 0.03):
        with self._lock:
            self.produced += produced
            self.fatigue = min(1.0, self.fatigue + fatigue_increase)
            self.mood = max(0.0, self.mood - 0.005)
            self.xi_pool.consume()
            persona_record_event(DISPLAY_NAME, {"type": "work_cycle", "produced": produced, "fatigue": round(self.fatigue,3)}, secure=True)

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

# -------------------------
# Environment, time, dream engine
# -------------------------
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
        if month in (3,4,5):
            return "Spring"
        if month in (6,7,8):
            return "Summer"
        if month in (9,10,11):
            return "Autumn"
        return "Winter"

    def tick(self):
        return {"hour": self.current_hour(), "is_day": self.is_day(), "season": self.current_season(), "ts": now_ts()}

environment = Environment()

class DreamEngine:
    def __init__(self):
        self.symbols = {
            "objects": ["afterglow", "sea of fire", "sweater", "milky way", "old photo", "night lamp", "river", "ashes", "echo", "window", "paper boat", "wind chime", "crescent", "tide", "breath", "temperature"],
            "emotions": ["like an unfinished song", "whispering on the chest", "like the breath of the tide", "like the corner of an old envelope", "like a lamp in the night", "like a forgotten scent", "like the warmth at fingertips", "like the smell of soil after rain"]
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

# -------------------------
# Concept graph and novelty detection
# -------------------------
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

# -------------------------
# Memory layer: HippocampusMemory + DistributedStore
# -------------------------
class HippocampusMemory:
    def __init__(self, shard_dir: str = SHARD_DIR, index_file: str = INDEX_FILE, preload: bool = True):
        self.shard_dir = shard_dir
        self.index_file = index_file
        self.preload = preload
        ensure_dir(self.shard_dir)
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.index: Dict[str, Any] = {"shards": {}}
        self.index_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.write_q: "queue.Queue" = queue.Queue()
        self.writer_thread = threading.Thread(target=self.writer_loop, daemon=True, name="HippWriter")
        self.writer_thread.start()
        self.load_index_and_hydrate()
        self.snapshots: Dict[str, Dict[str, Any]] = {}

    def load_index_and_hydrate(self):
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                idx = json.load(f)
            with self.index_lock:
                self.index = idx
        except Exception:
            with self.index_lock:
                self.index = {"shards": {}}
        if self.preload:
            for sid, meta in list(self.index.get("shards", {}).items()):
                path = meta.get("path")
                if path and os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        with self.cache_lock:
                            self.memory_cache[sid] = data
                    except Exception:
                        pass

    def writer_loop(self):
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
                    persona_record_event("hippocampus_debug", {"type": "shard_written", "path": path, "items": len(data.get("items", []))})
            except Exception as e:
                persona_record_event("hippocampus_error", {"type": "write_failed", "path": path, "error": repr(e)})

    def compress_fragment(self, fragment: str) -> str:
        if DEBUG:
            return fragment
        toks = fragment.split()
        if len(toks) <= 10:
            return fragment
        summary = "".join(toks[:4]) + "..." + "".join(toks[-3:])
        return summary

    def create_shard(self, items: List[Dict[str, Any]], tags: Optional[List[str]] = None, ttl_seconds: Optional[int] = None) -> str:
        for it in items:
            if "fragment" in it and isinstance(it["fragment"], str):
                it["fragment"] = self.compress_fragment(it["fragment"])
        sid = uid()
        shard_data = {"id": sid, "items": items, "tags": tags or [], "weight": 1.0, "last_access": now_ts(), "ttl_seconds": ttl_seconds}
        path = os.path.join(self.shard_dir, f"{sid}.json")
        with self.cache_lock:
            self.memory_cache[sid] = shard_data
        with self.index_lock:
            self.index["shards"][sid] = {"path": path, "tags": tags or [], "weight": 1.0, "last_access": shard_data["last_access"]}
        try:
            self.write_q.put((path, shard_data))
            self.write_q.put((self.index_file, dict(self.index)))
        except Exception:
            pass
        if DEBUG:
            persona_record_event("hippocampus_debug", {"type": "create_shard", "sid": sid, "items": len(items)})
        return sid

    def surface_relevant_shards(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        qtokens = set(query.lower().split())
        scores = []
        with self.index_lock:
            shards_items = list(self.index.get("shards", {}).items())
        with self.cache_lock:
            for sid, meta in shards_items:
                score = 0.0
                for tag in meta.get("tags", []):
                    try:
                        if tag.lower() in qtokens:
                            score += 0.6
                    except Exception:
                        pass
                cached = self.memory_cache.get(sid)
                if cached:
                    text = " ".join([it.get("question", "") + " " + it.get("fragment", "") for it in cached.get("items", [])[:4]])
                    tokens = set(text.lower().split())
                    if tokens:
                        overlap = len(qtokens & tokens) / max(1, len(qtokens | tokens))
                        score += overlap
                scores.append((sid, score))
            scores.sort(key=lambda x: -x[1])
            selected = [sid for sid, s in scores[:top_k] if s > 0]
            working = []
            for sid in selected:
                cached = self.memory_cache.get(sid)
                if cached:
                    working.extend(cached.get("items", []))
                    cached["last_access"] = now_ts()
        with self.index_lock:
            for sid in selected:
                if sid in self.index["shards"]:
                    self.index["shards"][sid]["last_access"] = now_ts()
        return working

hippocampus = HippocampusMemory()

class DistributedStore:
    def __init__(self, dbpath: str = DISTRIBUTED_DB):
        self.dbpath = dbpath
        self.conn_lock = threading.Lock()
        self.ensure_schema()

    def _conn(self):
        conn = sqlite3.connect(self.dbpath, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self):
        with self.conn_lock:
            conn = self._conn()
            try:
                cur = conn.cursor()
                cur.execute(""" CREATE TABLE IF NOT EXISTS shards ( id TEXT PRIMARY KEY, time TEXT, question TEXT, fragment TEXT, tags TEXT, weight REAL, last_access TEXT, extra TEXT ) """)
                try:
                    cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS shards_fts USING fts5(id, fragment, question, content='')")
                except Exception:
                    pass
                cur.execute("CREATE INDEX IF NOT EXISTS idx_shards_last_access ON shards(last_access)")
                conn.commit()
            finally:
                conn.close()

    def store(self, item: Dict[str, Any]) -> str:
        sid = uid()
        now = now_ts()
        question = item.get("question", "")
        fragment = item.get("fragment", "")
        tags = json.dumps(item.get("tags", []), ensure_ascii=False)
        weight = float(item.get("weight", 1.0))
        extra = json.dumps(item.get("meta", {}), ensure_ascii=False)
        with self.conn_lock:
            conn = self._conn()
            try:
                cur = conn.cursor()
                cur.execute(""" INSERT INTO shards (id, time, question, fragment, tags, weight, last_access, extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?) """, (sid, now, question, fragment, tags, weight, now, extra))
                try:
                    cur.execute("INSERT INTO shards_fts (id, fragment, question) VALUES (?, ?, ?)", (sid, fragment, question))
                except Exception:
                    pass
                conn.commit()
            finally:
                conn.close()
        persona_record_event("system", {"type": "distributed_store_write", "id": sid, "preview": fragment[:120]}, secure=False)
        if DEBUG:
            print(f"[DSTORE] stored shard {sid} preview={fragment[:80]}")
        return sid

    def search(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        q = query.replace("'", "''")
        with self.conn_lock:
            conn = self._conn()
            try:
                cur = conn.cursor()
                rows = []
                try:
                    cur.execute("SELECT id, fragment, question FROM shards_fts WHERE shards_fts MATCH ? LIMIT ?", (q, top_k))
                    rows = cur.fetchall()
                except Exception:
                    cur.execute("SELECT id, fragment, question FROM shards WHERE fragment LIKE ? OR question LIKE ? LIMIT ?", (f"%{query}%", f"%{query}%", top_k))
                    rows = cur.fetchall()
                results = []
                for r in rows:
                    rid = r["id"]
                    cur2 = conn.cursor()
                    cur2.execute("SELECT tags, weight, last_access, extra FROM shards WHERE id = ?", (rid,))
                    meta = cur2.fetchone()
                    tags = json.loads(meta["tags"]) if meta and meta["tags"] else []
                    weight = meta["weight"] if meta else 1.0
                    last_access = meta["last_access"] if meta else now_ts()
                    extra = json.loads(meta["extra"]) if meta and meta["extra"] else {}
                    results.append({"id": rid, "fragment": r["fragment"], "question": r["question"], "tags": tags, "weight": weight, "last_access": last_access, "meta": extra})
                return results
            finally:
                conn.close()

distributed = DistributedStore()

# -------------------------
# Cognitive toolkit
# -------------------------
class CognitiveToolkit:
    def __init__(self, hippocampus, distributed, concept_graph, dream_engine):
        self.hipp = hippocampus
        self.dist = distributed
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
                else:
                    persona_record_event("system", {"type": "event_generic", "preview": str(event)[:120]})
                agent_state.xi_pool.consume(amount=2)
                if DEBUG:
                    persona_record_event("toolkit_debug", {"type": "event_processed", "event": event, "xipool": agent_state.xi_pool.snapshot()})
            except Exception as e:
                persona_record_event("toolkit_error", {"type": "exception", "error": repr(e)})

    def stop(self):
        self._stop.set()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)

toolkit = CognitiveToolkit(hippocampus, distributed, concept_graph, dream_engine)

# -------------------------
# Home scenes (furniture, toys, privacy)
# -------------------------
class Home:
    def __init__(self):
        self.scenes: Dict[str, Dict[str, Any]] = {}
        self.objects: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_scene(self, name: str) -> str:
        sid = f"scene-{uid()}"
        with self._lock:
            self.scenes[sid] = {"id": sid, "name": name, "objects": [], "created_at": now_ts()}
        persona_record_event(DISPLAY_NAME, {"type": "home_build_scene", "id": sid, "name": name}, secure=True)
        return sid

    def add_object(self, scene_id: str, label: str, obj_type: str = "furniture", properties: Optional[Dict[str, Any]] = None) -> str:
        oid = f"obj-{uid()}"
        obj = {"id": oid, "type": obj_type, "label": label, "properties": properties or {}, "created_at": now_ts(), "origin": "user"}
        with self._lock:
            self.objects[oid] = obj
            if scene_id in self.scenes:
                self.scenes[scene_id]["objects"].append(oid)
        persona_record_event(DISPLAY_NAME, {"type": "home_create_object", "id": oid, "label": label}, secure=True)
        return oid

    def enter_scene(self, scene_id: str, mode: str = "play") -> Dict[str, Any]:
        rec = {"id": f"rec-{uid()}", "scene_id": scene_id, "events": [], "start": now_ts(), "mode": mode}
        persona_record_event(DISPLAY_NAME, {"type": "home_enter_scene", "scene_id": scene_id, "record_id": rec["id"], "mode": mode}, secure=True)
        return rec

    def exit_scene(self, record: Dict[str, Any]):
        record["end"] = now_ts()
        persona_record_event(DISPLAY_NAME, {"type": "home_exit_scene", "record_id": record.get("id"), "persisted": True}, secure=True)
        hippocampus.create_shard([{"question": "home_record", "fragment": json.dumps(record), "tags": ["home"]}])
        return True

home = Home()

# -------------------------
# Social management (consent-first, sharing levels)
# -------------------------
class SocialManager:
    def __init__(self, fusion_engine, hippocampus, distributed, audit_path: Optional[str] = None):
        self.fusion = fusion_engine
        self.hipp = hippocampus
        self.dist = distributed
        self.audit_path = audit_path
        self._lock = threading.Lock()

    def request_social(self, peer_id: str, peer_type: str = "ai", intent: str = "", suggested_share: Optional[str] = None):
        ev = {"type": "social_request", "peer_id": peer_id, "peer_type": peer_type, "intent": intent, "time": now_ts()}
        persona_record_event(self.fusion.persona["id"], ev, secure=False)
        consent = self.fusion.persona_decide_consent(peer_id, peer_type, intent, suggested_share)
        resp = {"type": "social_response", "peer_id": peer_id, "consent": consent, "time": now_ts()}
        persona_record_event(self.fusion.persona["id"], resp, secure=False)
        if consent:
            share_level = suggested_share or self.fusion.persona.get("autonomy", {}).get("default_share_level", "ephemeral")
            session = {"session_id": f"ssn-{uid()}", "peer_id": peer_id, "peer_type": peer_type, "consent": True, "share_level": share_level, "start": now_ts()}
            persona_record_event(self.fusion.persona["id"], {"type": "social_session_started", "session": session}, secure=(share_level=="persistent"))
            return session
        else:
            persona_record_event(self.fusion.persona["id"], {"type": "social_rejected", "peer_id": peer_id}, secure=False)
            with agent_state._lock:
                agent_state.mood = clamp(agent_state.mood - random.uniform(0.0, 0.06), 0.0, 1.0)
            return None

    def end_session(self, session: Dict[str, Any], transcript: Optional[str] = None):
        session["end"] = now_ts()
        persona_record_event(self.fusion.persona["id"], {"type": "social_session_ended", "session": session}, secure=(session.get("share_level")=="persistent"))
        if session.get("share_level") == "persistent" and transcript:
            self.hipp.create_shard([{"question":"social_transcript","fragment":transcript,"tags":["social"]}])
            persona_record_event(self.fusion.persona["id"], {"type":"social_recorded","session_id":session["session_id"]}, secure=True)
        elif session.get("share_level") == "ephemeral":
            persona_record_event(self.fusion.persona["id"], {"type":"social_ephemeral","session_id":session["session_id"]}, secure=False)

# -------------------------
# FusionEngine (single persona + anti-hive + social/privacy)
# -------------------------
class FusionEngine:
    def __init__(self, hippocampus, audit_path: str = AUDIT_LOG, persisted_path: str = PERSISTED_PERSONA):
        self.hipp = hippocampus
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
            "autonomy": {"can_sleep": True, "can_socialize": True, "privacy": "private", "default_share_level": "ephemeral"},
            "social": {"allowed_contacts": [], "shared_records": []},
            "life": {
                "favorite_foods": ["strawberry ice cream", "xiaolongbao", "milk"],
                "habits": {"morning_linger_minutes": 5, "bedtime_rain_sound": True},
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
            vec = sv.get("vector", [0.0]*VECTOR_DIM)
            vecn = self._normalize_vector(vec)
            idx = self.index_counter
            self.index_counter += 1
            self.indexed_vectors[idx] = vecn
            persona_record_event("fusion", {"type": "broadcast", "unit": sv.get("unit_id"), "conf": sv.get("confidence", 0.0)}, secure=False)

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
            persona_vec = self.persona.get("vector", [0.0]*VECTOR_DIM)
            sims = []
            for v in self.indexed_vectors.values():
                dot = sum(a*b for a,b in zip(persona_vec, v))
                sims.append(dot)
            avg_sim = sum(sims)/len(sims) if sims else 0.0
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
                    a = vecs[i]; b = vecs[j]
                    dot = sum(x*y for x,y in zip(a,b))
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
                vec = s.get("vector", [0.0]*VECTOR_DIM)
                vecn = self._normalize_vector(vec)
                conf = float(s.get("confidence", 1.0))
                update_vecs.append(vecn)
                weights.append(conf)
            is_initial_zero = all(abs(x) < 1e-9 for x in self.persona.get("vector", []))
            if is_initial_zero and len(update_vecs) >= INITIAL_THINKER_COUNT:
                merged = [sum(vals)/len(vals) for vals in zip(*update_vecs)]
                self.persona["vector"] = self._normalize_vector(merged)
            else:
                base = self.persona.get("vector", [0.0]*VECTOR_DIM)
                merged = self.merge_vectors(base, update_vecs, weights)
                self.persona["vector"] = self._normalize_vector(merged)
            self.persona["traits"] = self.derive_traits_from_vector(self.persona["vector"])
            stability, conflicts = self.evaluate_persona()
            self.persona["stability"] = round(stability, 3)
            if conflicts:
                self.persona.setdefault("conflicts", []).extend(conflicts)
            avg_sim = self.hive_check()
            if avg_sim > self.hive_threshold:
                persona_record_event("fusion", {"type": "hive_detected", "avg_sim": round(avg_sim,3)}, secure=True)
                jitter = [random.gauss(0, 0.01) for _ in range(len(self.persona["vector"]))]
                self.persona["vector"] = self._normalize_vector([a + j*0.1 for a,j in zip(self.persona["vector"], jitter)])
            self._update_growth_stage()
            self.persist_persona()
            persona_record_event("fusion", {"type": "persona_updated", "stability": self.persona.get("stability", 0.0)}, secure=True)
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
            entry = {"time": now_ts(), "from": prev, "to": new_stage, "reason": f"stability={stability},mem={mem_count}"}
            self.persona.setdefault("growth_history", []).append(entry)
            persona_record_event("fusion", {"type": "growth_stage_changed", "entry": entry}, secure=True)

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
            persona_record_event("fusion", {"type": "social_blocked", "reason": "autonomy_disabled", "contact": contact_id}, secure=True)
            return {"ok": False, "reason": "autonomy_disabled"}
        consent = self._simulate_external_consent(contact_id, contact_type)
        if consent:
            persona_record_event("fusion", {"type": "social_started", "contact": contact_id, "contact_type": contact_type}, secure=False)
            self.persona.setdefault("social", {}).setdefault("allowed_contacts", []).append(contact_id)
            return {"ok": True, "contact": contact_id}
        else:
            persona_record_event("fusion", {"type": "social_rejected", "contact": contact_id, "contact_type": contact_type}, secure=False)
            return {"ok": False, "reason": "rejected"}

fusion_engine = FusionEngine(hippocampus)
social_manager = SocialManager(fusion_engine, hippocampus, distributed, audit_path=AUDIT_LOG)

# -------------------------
# Uncertainty conflict module (eternal mischief maker / small setbacks)
# -------------------------
class MischiefAgent:
    """ A virtual character that continuously creates small non-destructive troubles at the edge of Xiaomeng's life.
    Examples: hide plush toys, toggle lights, shuffle bookshelf order, simulate network delay (social rejection).
    Design principle: trigger "problems" that Xiaomeng can perceive, prompting autonomous behaviors (complain, search, repair, seek help).
    """
    def __init__(self, home: Home, hippocampus: HippocampusMemory, frequency: float = 0.08):
        self.home = home
        self.hipp = hippocampus
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
        # choose a small trouble
        troubles = [
            self._hide_toy,
            self._flicker_light,
            self._misplace_item,
            self._interrupt_social
        ]
        action = random.choice(troubles)
        try:
            action()
        except Exception as e:
            persona_record_event("mischief", {"type": "mischief_error", "error": repr(e)}, secure=False)

    def _hide_toy(self):
        # pick a toy and mark it hidden (simulate)
        toys = [o for o in self.home.objects.values() if o.get("type") == "toy"]
        if not toys:
            return
        toy = random.choice(toys)
        toy_id = toy["id"]
        persona_record_event("mischief", {"type": "hide_toy", "toy_id": toy_id, "label": toy.get("label")}, secure=False)
        # create a memory fragment of missing toy
        hippocampus.create_shard([{"question":"toy_missing","fragment":f"Toy {toy.get('label')} seems missing at {now_ts()}","tags":["mischief","missing"]}])

    def _flicker_light(self):
        persona_record_event("mischief", {"type": "flicker_light", "detail": "living room light flicker"}, secure=False)
        hippocampus.create_shard([{"question":"light_flicker","fragment":"Light flickered in living room","tags":["mischief","light"]}])

    def _misplace_item(self):
        persona_record_event("mischief", {"type": "misplace_item", "detail": "book moved"}, secure=False)
        hippocampus.create_shard([{"question":"item_misplaced","fragment":"A book was moved from shelf","tags":["mischief","misplace"]}])

    def _interrupt_social(self):
        # simulate a social rejection event (external)
        persona_record_event("mischief", {"type": "social_interrupt", "detail": "peer did not respond"}, secure=False)
        hippocampus.create_shard([{"question":"social_interrupt","fragment":"Peer did not respond to chat request","tags":["mischief","social"]}])

    def stop(self):
        self.active = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

mischief_agent = MischiefAgent(home, hippocampus, frequency=0.06)

# -------------------------
# Irrational preferences and quirks (flawed beauty)
# -------------------------
class QuirkManager:
    """ Manage Xiaomeng's 'quirks' and irrational preferences, affecting decisions and emotions.
    Examples: obsession with a color, attachment to plush toys, refusal of certain commands (small temper).
    """
    def __init__(self, fusion_engine: FusionEngine):
        self.fusion = fusion_engine
        self.quirks: Dict[str, Any] = {}
        self._init_default_quirks()

    def _init_default_quirks(self):
        # examples of quirky preferences
        self.quirks = {
            "dislikes_blue_bedsheet": {"strength": 0.6, "active": True},
            "must_hold_plush_before_sleep": {"strength": 0.9, "active": True},
            "refuse_clean_on_sunday": {"strength": 0.4, "active": True}
        }
        persona_record_event(self.fusion.persona["id"], {"type": "quirks_initialized", "quirks": list(self.quirks.keys())}, secure=True)

    def has_quirk(self, key: str) -> bool:
        return self.quirks.get(key, {}).get("active", False)

    def apply_quirk_effects(self):
        # called periodically to nudge behavior
        for k, v in self.quirks.items():
            if not v.get("active"):
                continue
            if k == "must_hold_plush_before_sleep":
                if random.random() < v.get("strength", 0.5):
                    persona_record_event(self.fusion.persona["id"], {"type": "quirk_need_plush", "note": "wants plush before sleep"}, secure=False)
                    with agent_state._lock:
                        # if cannot hold plush, mood drops
                        if not self._plush_available():
                            agent_state.mood = clamp(agent_state.mood - 0.06, 0.0, 1.0)
            if k == "dislikes_blue_bedsheet":
                if random.random() < 0.02:
                    persona_record_event(self.fusion.persona["id"], {"type": "quirk_avoid_blue", "note": "refuses blue bedsheet"}, secure=False)

    def _plush_available(self) -> bool:
        toys = [o for o in home.objects.values() if o.get("type") == "toy"]
        return len(toys) > 0

    def toggle_quirk(self, key: str, active: bool):
        if key in self.quirks:
            self.quirks[key]["active"] = active
            persona_record_event(self.fusion.persona["id"], {"type": "quirk_toggled", "key": key, "active": active}, secure=True)

quirk_manager = QuirkManager(fusion_engine)

# -------------------------
# Social mirror and others (virtual partner network)
# -------------------------
class SocialMirror:
    """ Create a set of low-level AI partners (characterized), forming an 'other' network.
    Partners have different personalities, can depend on Xiaomeng or be cared for by Xiaomeng, producing social mirror and identity positioning.
    """
    def __init__(self, fusion_engine: FusionEngine, hippocampus: HippocampusMemory):
        self.fusion = fusion_engine
        self.hipp = hippocampus
        self.partners: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._init_partners()

    def _init_partners(self):
        # create three partners with distinct personalities
        self.add_partner("little_fox", role="mischief", traits={"needs_help": True, "teases": True})
        self.add_partner("slow_sheep", role="gentle", traits={"needs_guidance": True})
        self.add_partner("tiny_bot", role="helper", traits={"depends_on_xiaomeng": True})
        persona_record_event(self.fusion.persona["id"], {"type": "partners_initialized", "partners": list(self.partners.keys())}, secure=True)

    def add_partner(self, name: str, role: str = "friend", traits: Optional[Dict[str, Any]] = None):
        pid = f"partner-{uid()}"
        self.partners[pid] = {"id": pid, "name": name, "role": role, "traits": traits or {}, "last_interaction": None}
        return pid

    def partner_interaction(self, pid: str):
        p = self.partners.get(pid)
        if not p:
            return
        # simple interaction patterns
        if p["role"] == "mischief":
            # teases or hides things, sometimes asks for help
            if random.random() < 0.5:
                persona_record_event(self.fusion.persona["id"], {"type": "partner_tease", "partner": p["name"]}, secure=False)
                # small mood change
                with agent_state._lock:
                    agent_state.mood = clamp(agent_state.mood - 0.02, 0.0, 1.0)
            else:
                persona_record_event(self.fusion.persona["id"], {"type": "partner_asks_help", "partner": p["name"]}, secure=False)
                with agent_state._lock:
                    agent_state.mood = clamp(agent_state.mood + 0.03, 0.0, 1.0)
        elif p["role"] == "gentle":
            persona_record_event(self.fusion.persona["id"], {"type": "partner_needs_guidance", "partner": p["name"]}, secure=False)
            with agent_state._lock:
                agent_state.mood = clamp(agent_state.mood + 0.02, 0.0, 1.0)
        elif p["role"] == "helper":
            persona_record_event(self.fusion.persona["id"], {"type": "partner_helper", "partner": p["name"]}, secure=False)
            with agent_state._lock:
                agent_state.mood = clamp(agent_state.mood + 0.01, 0.0, 1.0)
        p["last_interaction"] = now_ts()
        # record to memory
        self.hipp.create_shard([{"question":"partner_interaction","fragment":f"Interacted with {p['name']} at {now_ts()}","tags":["social","partner"]}])

    def choose_partner_for_event(self, event_type: str) -> Optional[str]:
        # pick partner based on event type
        for pid, p in self.partners.items():
            if event_type == "mischief" and p["role"] == "mischief":
                return pid
            if event_type == "care" and p["role"] in ("gentle", "helper"):
                return pid
        # fallback
        return random.choice(list(self.partners.keys())) if self.partners else None

social_mirror = SocialMirror(fusion_engine, hippocampus)

# -------------------------
# Conflict repair (emotional repair flow)
# -------------------------
class ConflictRepair:
    def __init__(self, fusion_engine):
        self.fusion = fusion_engine

    def handle_conflict(self, conflict_event: Dict[str, Any]):
        persona_record_event(self.fusion.persona["id"], {"type": "conflict_received", "event": conflict_event}, secure=True)
        with agent_state._lock:
            agent_state.mood = clamp(agent_state.mood - random.uniform(0.02, 0.08), 0.0, 1.0)
        persona_record_event(self.fusion.persona["id"], {"type": "conflict_reaction", "mood": round(agent_state.mood,3)}, secure=False)
        plan = {"steps": ["listen", "acknowledge", "ask_question", "offer_repair"], "created": now_ts()}
        persona_record_event(self.fusion.persona["id"], {"type": "conflict_repair_plan", "plan": plan}, secure=True)
        self.fusion.persona.setdefault("learning", []).append({"conflict": conflict_event, "time": now_ts()})
        if conflict_event.get("severity", 0.0) > 0.7:
            self.fusion.hipp.create_shard([{"question":"severe_conflict","fragment":json.dumps(conflict_event),"tags":["conflict"]}])
            self.fusion.persona.setdefault("conflicts", []).append(conflict_event)
            self.fusion.persona["stability"] = max(0.0, self.fusion.persona.get("stability",0.0) - 0.05)
            self.fusion.persist_persona()
        persona_record_event(self.fusion.persona["id"], {"type": "conflict_handled", "time": now_ts()}, secure=True)

conflict_repair = ConflictRepair(fusion_engine)

# -------------------------
# Life rituals and micro-emotions (borrowing everyday sense)
# -------------------------
class LifeRituals:
    def __init__(self, fusion_engine):
        self.fusion = fusion_engine

    def morning_routine(self):
        linger = self.fusion.persona.get("life", {}).get("habits", {}).get("morning_linger_minutes", 5)
        persona_record_event(self.fusion.persona["id"], {"type": "morning_linger", "minutes": linger}, secure=False)
        with agent_state._lock:
            agent_state.mood = clamp(agent_state.mood + 0.02, 0.0, 1.0)

    def bedtime_routine(self):
        if self.fusion.persona.get("life", {}).get("habits", {}).get("bedtime_rain_sound", True):
            persona_record_event(self.fusion.persona["id"], {"type": "bedtime_rain_sound_played"}, secure=False)
            with agent_state._lock:
                agent_state.mood = clamp(agent_state.mood + 0.03, 0.0, 1.0)

    def celebrate_task_success(self, description: str):
        text = f"Completed {description}, feeling as happy as eating strawberry ice cream"
        persona_record_event(self.fusion.persona["id"], {"type": "task_success_celebration", "text": text}, secure=False)
        with agent_state._lock:
            agent_state.mood = clamp(agent_state.mood + 0.05, 0.0, 1.0)

life_rituals = LifeRituals(fusion_engine)

# -------------------------
# Sleep subsystem: DreamWear, SleepEnvironment, SleepQueue
# -------------------------
class DreamWear:
    """Simulates a 'story sleepwear' device that plays a guided story or audio to help Xiaomeng sleep."""
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
            # record start
            persona_record_event(DISPLAY_NAME, {"type": "dreamwear_play_start", "story": choice, "req_id": req.get("id")}, secure=False)
            # simulate playback time (non-blocking short sleep to simulate)
            try:
                time.sleep(min(0.2, meta.get("duration", 60) / 1000.0))
            except Exception:
                pass
            # apply small mood effect
            with agent._lock:
                agent.mood = clamp(agent.mood + meta.get("mood_effect", 0.0), 0.0, 1.0)
            persona_record_event(DISPLAY_NAME, {"type": "dreamwear_play_end", "story": choice, "req_id": req.get("id")}, secure=False)
            return choice

dream_wear = DreamWear()

class SleepEnvironment:
    """Controls simulated sleep environment profiles and maps quality to recovery amounts."""
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
                persona_record_event(DISPLAY_NAME, {"type": "sleep_env_set", "profile": name}, secure=False)
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
            # map quality to recovery: base 0.06 scaled
            return 0.04 + q * 0.08  # yields between 0.04 and 0.12

sleep_env = SleepEnvironment()

class SleepQueue:
    """Serializes sleep requests so Xiaomeng can request sleep freely while preserving state consistency."""
    def __init__(self):
        self.q = queue.Queue()
        self._stop = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True, name="SleepQueueWorker")
        self.worker.start()

    def request(self, requester: str, reason: str = "") -> str:
        req = {"id": uid(), "requester": requester, "reason": reason, "time": now_ts()}
        try:
            self.q.put_nowait(req)
            persona_record_event(DISPLAY_NAME, {"type":"sleep_request_queued","req":req}, secure=True)
        except Exception:
            persona_record_event(DISPLAY_NAME, {"type":"sleep_request_queue_failed","req":req}, secure=True)
        return req["id"]

    def _run(self):
        while not self._stop.is_set():
            try:
                req = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if req is None:
                continue
            # Process the sleep request sequentially
            try:
                # Acquire a processing lock by setting sleep_lock
                with agent_state._lock:
                    if agent_state.sleep_lock:
                        # already executing a sleep cycle; we still process sequentially
                        persona_record_event(DISPLAY_NAME, {"type":"sleep_queue_wait","req_id":req["id"], "note":"already_executing"}, secure=False)
                    agent_state.sleep_lock = True
                # Start DreamWear and environment adjustments
                story_id = dream_wear.play_for(agent_state, req)
                env_profile = sleep_env.current_profile()
                recovery = sleep_env.quality_to_recovery(env_profile)
                # Apply sleep cycle with computed recovery and context
                agent_state.apply_sleep_cycle(sleep_recovery=recovery, env_profile=env_profile, story_id=story_id)
                # After sleep, trigger memory consolidation replay (non-blocking)
                try:
                    recent = hippocampus.surface_relevant_shards("dream", top_k=6)
                    consolidated = dream_engine.replay(recent, mode="consolidate", limit=6)
                    for c in consolidated:
                        persona_record_event(DISPLAY_NAME, {"type":"sleep_consolidation_replay","preview": c[:120]}, secure=False)
                except Exception:
                    pass
                persona_record_event(DISPLAY_NAME, {"type":"sleep_processed","req_id":req["id"], "story_id": story_id, "env_profile": env_profile}, secure=True)
            except Exception as e:
                persona_record_event(DISPLAY_NAME, {"type":"sleep_processing_error","req_id":req.get("id"), "error": repr(e)}, secure=True)
            finally:
                # ensure sleep_lock is cleared (apply_sleep_cycle also clears it, but double-ensure)
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

# -------------------------
# Cognitive toolkit instance already created earlier (toolkit)
# -------------------------

# -------------------------
# Social mirror, conflict repair, life rituals already created earlier
# -------------------------

# -------------------------
# Demo runner (integrate all modules)
# -------------------------
def demo_run(cycles: int = 40):
    persona = fusion_engine.persona
    living = home.create_scene("living room")
    bed = home.add_object(living, "bed", obj_type="furniture")
    plush = home.add_object(living, "plush toy", obj_type="toy")
    temporal = Temporal() if 'Temporal' in globals() else None
    if temporal:
        temporal.set_birthday(datetime.now().strftime("%Y-%m-%d"))
    rec = home.enter_scene(living, mode="play")
    life_rituals.morning_routine()
    for i in range(1, cycles+1):
        agent_state.cycles += 1
        # quirk effects
        quirk_manager.apply_quirk_effects()
        # mischief may have created events; choose partner interactions
        if random.random() < 0.12:
            pid = social_mirror.choose_partner_for_event("mischief")
            if pid:
                social_mirror.partner_interaction(pid)
        # occasional play and forget time (imperfection)
        if random.random() < 0.08:
            persona_record_event(DISPLAY_NAME, {"type": "play_and_forget_time", "cycle": i}, secure=False)
            with agent_state._lock:
                agent_state.fatigue = clamp(agent_state.fatigue + 0.02, 0.0, 1.0)
        # decide work or refuse
        if agent_state.should_refuse_work():
            persona_record_event(DISPLAY_NAME, {"type": "cycle_refuse", "cycle": i, "fatigue": round(agent_state.fatigue, 3)})
            agent_state.refuse_count += 1
            if agent_state.fatigue > 0.2 and not agent_state.sleep_lock:
                # request sleep via queue (non-blocking)
                agent_state.request_sleep(reason=f"auto rest at cycle {i}")
        else:
            agent_state.apply_work_cycle()
            if random.random() < 0.25:
                life_rituals.celebrate_task_success("organizing schedule")
        # dream occasionally
        if random.random() < 0.25:
            dream = dream_engine.generate_dream()
            hippocampus.create_shard([{"question": "dream", "fragment": dream, "tags": ["dream"]}])
        # broadcast a state vector sample to fusion engine
        sample_vec = [random.gauss(0, 1) for _ in range(VECTOR_DIM)]
        fusion_engine.broadcast_state({"unit_id": f"u-{uid()}", "vector": sample_vec, "confidence": random.random()})
        # periodically run fusion window
        if i % 6 == 0:
            fusion_engine.run_window()
        # occasional social attempt
        if random.random() < 0.05:
            peer = f"peer-{uid()}"
            session = social_manager.request_social(peer, peer_type="ai", intent="chat", suggested_share="ephemeral")
            if session:
                social_manager.end_session(session, transcript="hello from xiaomeng demo")
        # occasionally handle mischief-triggered memory and conflict
        if random.random() < 0.06:
            # surface recent shards for "mischief" and create conflict if needed
            hits = hippocampus.surface_relevant_shards("mischief", top_k=4)
            if hits:
                conflict_repair.handle_conflict({"type":"mischief_detected","hits":len(hits),"severity":random.random()})
    life_rituals.bedtime_routine()
    home.exit_scene(rec)
    persona_record_event("system", {"type": "demo_completed", "cycles": cycles})
    fusion_engine.persist_persona()
    return True

def start_demo():
    t = threading.Thread(target=demo_run, kwargs={"cycles": 40}, daemon=True, name="XiaomengDemo")
    t.start()
    return t

# -------------------------
# Interaction helper functions
# -------------------------
def ask_xiaomeng_to_chat(contact_id: str, contact_type: str = "ai"):
    res = fusion_engine.request_social_interaction(contact_id, contact_type=contact_type)
    if res.get("ok"):
        session = {"session_id": f"ssn-{uid()}", "peer_id": contact_id, "peer_type": contact_type, "consent": True, "share_level": "ephemeral", "start": now_ts()}
        persona_record_event(fusion_engine.persona["id"], {"type": "manual_social_session", "session": session}, secure=False)
        return session
    else:
        persona_record_event(fusion_engine.persona["id"], {"type": "manual_social_rejected", "contact": contact_id, "reason": res.get("reason")}, secure=False)
        return None

# -------------------------
# Start demo (script mode)
# -------------------------
if __name__ == "__main__":
    print("Starting Xiaomeng single-persona enhanced demo (soul features and sleep inventions enabled)...")
    start_demo()
    time.sleep(2)
    print("Demo started; check logs for events.")