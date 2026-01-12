#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qem_autonomous_final_embedded.py
Ultimate improved single-file — QEM Autonomous Final (with quarantine sublimation retry embedded)
  - Based on qem_autonomous_final, embeds retry_quarantine_and_sublimate
  - Automatically retries top-K pairs from quarantine after a single run (configurable)
  - Supports --debug-run-once, --relax-sim, --force-cluster, --cluster-eps
  - Writes detailed debug logs to qem_cloud_data/debug_log.txt

Notes:
  - Place this file in the project root and run: python qem_autonomous_final_embedded.py [--debug-run-once]
  - To force cluster merges: --force-cluster --cluster-eps 0.25
  - Environment variables can override some parameters (see Config)
"""
from __future__ import annotations
import os
import sys
import time
import json
import uuid
import math
import random
import threading
import traceback
import shutil
import http.server
import socketserver
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Optional acceleration
try:
    import numpy as np
except Exception:
    np = None

# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    base_dir: str = os.path.abspath(".")
    data_dir: str = os.path.join(base_dir, "qem_cloud_data")
    shard_dir: str = field(init=False)
    ent_path: str = field(init=False)
    seed_path: str = field(init=False)
    ledger_path: str = field(init=False)
    comp_log: str = field(init=False)
    backup_dir: str = field(init=False)
    debug_log: str = field(init=False)
    quarantine_path: str = field(init=False)
    auto_disable_file: str = field(init=False)
    shutdown_signal: str = field(init=False)

    dim: int = int(os.environ.get("QEM_DIM", 64))
    default_sim: float = float(os.environ.get("QEM_SIM", 0.70))
    default_iters: int = int(os.environ.get("QEM_ITERS", 4))
    quant_bits: int = int(os.environ.get("QEM_QUANT_BITS", 8))
    min_group: int = int(os.environ.get("QEM_MIN_GROUP", 2))
    freq_alpha: float = float(os.environ.get("QEM_FREQ_ALPHA", 1.0))
    freq_beta: float = float(os.environ.get("QEM_FREQ_BETA", 1.0))
    pair_sim_factor: float = float(os.environ.get("QEM_PAIR_SIM_FACTOR", 1.0))
    status_port: int = int(os.environ.get("QEM_STATUS_PORT", 0))
    max_auto_inject: int = int(os.environ.get("QEM_MAX_AUTO_INJECT", 128))
    poll_interval: float = float(os.environ.get("QEM_POLL_INTERVAL", 5.0))
    idle_run_seconds: int = int(os.environ.get("QEM_IDLE_RUN_SECONDS", 120))
    log_prefix: str = "[QEM-Auto-Final-Embedded]"
    sim_min: float = float(os.environ.get("QEM_SIM_MIN", 0.35))
    quarantine_retry_limit: int = int(os.environ.get("QEM_QUARANTINE_RETRY", 3))
    max_backup_keep: int = int(os.environ.get("QEM_MAX_BACKUP_KEEP", 10))

    def __post_init__(self):
        self.shard_dir = os.path.join(self.data_dir, "shards")
        self.ent_path = os.path.join(self.data_dir, "ents.json")
        self.seed_path = os.path.join(self.data_dir, "seeds.json")
        self.ledger_path = os.path.join(self.data_dir, "ledger.json")
        self.comp_log = os.path.join(self.data_dir, "compression_log.json")
        self.backup_dir = os.path.join(self.data_dir, "backups")
        self.debug_log = os.path.join(self.data_dir, "debug_log.txt")
        self.quarantine_path = os.path.join(self.data_dir, "quarantine.json")
        self.auto_disable_file = os.path.join(self.data_dir, "auto_run.disable")
        self.shutdown_signal = os.path.join(self.data_dir, "shutdown.signal")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.shard_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

cfg = Config()

# -------------------------
# Utilities & logging
# -------------------------
def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def uid(prefix: str = "") -> str:
    return prefix + str(uuid.uuid4())[:12]

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

def _append_debug(s: str):
    try:
        with open(cfg.debug_log, "a", encoding="utf-8") as f:
            f.write(f"{now_ts()} {s}\n")
    except Exception:
        pass

def log(msg: str):
    line = f"{cfg.log_prefix} [{now_ts()}] {msg}"
    print(line, flush=True)
    _append_debug(msg)

def log_exc(prefix: str = "EXC"):
    tb = traceback.format_exc()
    log(f"{prefix}: {tb}")

# -------------------------
# Ledger & comp log
# -------------------------
class Ledger:
    _lock = threading.RLock()
    chain: List[Dict[str, Any]] = []

    @classmethod
    def record(cls, op: str, obj_id: str, info: Dict[str, Any]):
        with cls._lock:
            prev = cls.chain[-1]['hash'] if cls.chain else ''
            entry = {"ts": now_ts(), "op": op, "id": obj_id, "info": info, "prev": prev}
            try:
                s = json.dumps(entry, sort_keys=True, ensure_ascii=False)
                import hashlib
                entry['hash'] = hashlib.sha256(s.encode('utf-8')).hexdigest()
            except Exception:
                entry['hash'] = uid("h-")
            cls.chain.append(entry)

    @classmethod
    def dump(cls):
        with cls._lock:
            try:
                safe_write_json(cfg.ledger_path, cls.chain)
                log(f"Ledger dumped to {cfg.ledger_path}")
            except Exception:
                log_exc("Ledger.dump failed")

def append_comp_log(rec: Dict[str, Any]):
    arr = []
    try:
        if os.path.exists(cfg.comp_log):
            with open(cfg.comp_log, "r", encoding="utf-8") as f:
                arr = json.load(f)
    except Exception:
        arr = []
    arr.append(rec)
    safe_write_json(cfg.comp_log, arr)

# -------------------------
# Frequency store
# -------------------------
class FrequencyStore:
    def __init__(self, path=os.path.join(cfg.data_dir, "freq.json")):
        self.path = path
        self.lock = threading.RLock()
        self.counts: Dict[str,int] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.counts = json.load(f)
            except Exception:
                self.counts = {}

    def inc(self, k:str, d:int=1):
        with self.lock:
            self.counts[k] = self.counts.get(k,0) + d
            if self.counts[k] % 50 == 0:
                safe_write_json(self.path, self.counts)

    def get(self,k:str)->int:
        with self.lock:
            return self.counts.get(k,0)

    def snapshot(self):
        with self.lock:
            return dict(self.counts)

freq_store = FrequencyStore()

# -------------------------
# Data models
# -------------------------
@dataclass
class EntNode:
    id: str
    vec: Optional[List[float]]
    shards: List[str]
    score: float = 1.0
    ts: float = field(default_factory=time.time)

    def to_dict(self):
        return {"id":self.id,"vec":self.vec,"shards":self.shards,"score":self.score,"ts":self.ts}

    @staticmethod
    def from_dict(d):
        return EntNode(id=d["id"], vec=d.get("vec"), shards=d.get("shards",[]), score=d.get("score",1.0), ts=d.get("ts", time.time()))

@dataclass
class SeedNode:
    id: str
    seed_vec: Optional[List[float]]
    members: List[str]
    diffs: Dict[str, List[int]] = field(default_factory=dict)
    quant_meta: Dict[str,Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self):
        return {"id":self.id,"seed_vec":self.seed_vec,"members":self.members,"diffs":self.diffs,"quant_meta":self.quant_meta,"ts":self.ts}

    @staticmethod
    def from_dict(d):
        return SeedNode(id=d["id"], seed_vec=d.get("seed_vec"), members=d.get("members",[]), diffs=d.get("diffs",{}), quant_meta=d.get("quant_meta",{}), ts=d.get("ts", time.time()))

# -------------------------
# Registry & SeedIndex
# -------------------------
class EntRegistry:
    def __init__(self, path=cfg.ent_path):
        self.path = path
        self.lock = threading.RLock()
        self.nodes: Dict[str,EntNode] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path,"r",encoding="utf-8") as f:
                    data = json.load(f)
                for nid, nd in data.get("nodes",{}).items():
                    self.nodes[nid] = EntNode.from_dict(nd)
            except Exception:
                self.nodes = {}

    def save(self):
        with self.lock:
            data = {"nodes": {nid:n.to_dict() for nid,n in self.nodes.items()}}
            safe_write_json(self.path, data)

    def register(self,node:EntNode):
        with self.lock:
            self.nodes[node.id] = node
            try:
                safe_write_json(self.path, {"nodes": {nid:n.to_dict() for nid,n in self.nodes.items()}})
            except Exception:
                pass
        Ledger.record("ENT_REGISTER", node.id, {"shards": len(node.shards)})

class SeedIndex:
    def __init__(self, path=cfg.seed_path):
        self.path = path
        self.lock = threading.RLock()
        self.seeds: Dict[str,SeedNode] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path,"r",encoding="utf-8") as f:
                    data = json.load(f)
                for sid, sd in data.get("seeds",{}).items():
                    self.seeds[sid] = SeedNode.from_dict(sd)
            except Exception:
                self.seeds = {}

    def save(self):
        with self.lock:
            data = {"seeds": {sid:s.to_dict() for sid,s in self.seeds.items()}}
            safe_write_json(self.path, data)

    def register(self, seed:SeedNode):
        with self.lock:
            self.seeds[seed.id] = seed
            try:
                safe_write_json(self.path, {"seeds": {sid:s.to_dict() for sid,s in self.seeds.items()}})
            except Exception:
                pass
        Ledger.record("SEED_REGISTER", seed.id, {"members": len(seed.members)})

# -------------------------
# Vector helpers
# -------------------------
def _cosine(a,b):
    if not a or not b:
        return 0.0
    try:
        if np is not None:
            aa = np.array(a, dtype=float); bb = np.array(b, dtype=float)
            an = np.linalg.norm(aa) + 1e-12; bn = np.linalg.norm(bb) + 1e-12
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

# -------------------------
# Quantization helpers
# -------------------------
def quantize_list(vecs: List[List[float]], bits:int=cfg.quant_bits):
    flat = [x for v in vecs for x in v] if vecs else []
    if not flat:
        return [], {}
    mn = min(flat); mx = max(flat)
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
    mn = meta.get("min", 0.0); mx = meta.get("max", 0.0); bits = meta.get("bits", cfg.quant_bits)
    levels = (1 << bits) - 1
    if levels == 0:
        return [mn for _ in qvec]
    return [mn + (x / levels) * (mx - mn) for x in qvec]

# -------------------------
# Compressor with both strategies
# -------------------------
class Compressor:
    def __init__(self, registry:EntRegistry, seed_index:SeedIndex):
        self.registry = registry
        self.seed_index = seed_index

    # greedy cluster used by build_seeds and force-cluster
    def greedy_cluster(self, nodes:List[EntNode], sim_thresh:float, min_group:int):
        groups = []; used = set()
        for i, a in enumerate(nodes):
            if a.id in used:
                continue
            group = [a]; used.add(a.id)
            for b in nodes[i+1:]:
                if b.id in used:
                    continue
                try:
                    if _cosine(a.vec, b.vec) >= sim_thresh:
                        group.append(b); used.add(b.id)
                except Exception:
                    continue
            if len(group) >= min_group:
                groups.append(group)
        return groups

    def build_seeds(self, sim_thresh:float=cfg.default_sim, min_group:int=cfg.min_group, quant_bits:int=cfg.quant_bits):
        nodes = list(self.registry.nodes.values())
        if not nodes:
            return {"created": 0}
        groups = self.greedy_cluster(nodes, sim_thresh, min_group)
        created = 0
        for g in groups:
            created += self._create_seed(g, quant_bits)
        return {"created": created, "groups": len(groups)}

    def _create_seed(self, group:List[EntNode], quant_bits:int):
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
        seed_node = SeedNode(id=uid("seed-"), seed_vec=seed_vec, members=member_ids, diffs=diffs_map, quant_meta=meta)
        self.seed_index.register(seed_node)
        with self.registry.lock:
            for n in group:
                self.registry.nodes.pop(n.id, None)
            self.registry.save()
        Ledger.record("SEED_CREATED", seed_node.id, {"from":[n.id for n in group], "members":len(member_ids)})
        append_comp_log({"ts": now_ts(), "seed": seed_node.id, "from":[n.id for n in group], "members": len(member_ids)})
        log(f"Created seed {seed_node.id} from {[n.id for n in group]}")
        return 1

    # Method A: sign-agnostic + low-sim sublimation + quarantine
    def complementary_sublimate_flexible(self, sim_thresh:float=cfg.default_sim, sim_min:float=cfg.sim_min, allow_sign_flip:bool=True, alpha:float=cfg.freq_alpha, beta:float=cfg.freq_beta, quant_bits:int=cfg.quant_bits, max_iters:int=cfg.default_iters, target_nodes:Optional[int]=None):
        quarantine_path = cfg.quarantine_path

        def load_quarantine():
            try:
                if os.path.exists(quarantine_path):
                    return json.load(open(quarantine_path, "r", encoding="utf-8"))
            except Exception:
                pass
            return []

        def save_quarantine(q):
            try:
                safe_write_json(quarantine_path, q)
            except Exception:
                pass

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
            return sim * (abs(sa) + abs(sb) + 1e-6) * sign_term * sign_bonus * cfg.pair_sim_factor

        quarantine = load_quarantine()
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
            candidate_list = []
            for i, a in enumerate(nodes_sorted):
                if a.id in used:
                    continue
                sa = scores.get(a.id, 0.0)
                best = None; best_metric = 0.0; best_sim = 0.0
                for b in nodes_sorted[i+1:]:
                    if b.id in used:
                        continue
                    sb = scores.get(b.id, 0.0)
                    sim = _cosine(a.vec, b.vec) if (a.vec and b.vec) else 0.0
                    m = pair_metric(a, b, sa, sb)
                    candidate_list.append((a.id, b.id, sim, sa, sb, m))
                    if sim >= sim_thresh and m > best_metric:
                        best_metric = m; best = b; best_sim = sim
                    elif best is None and m > best_metric:
                        best_metric = m; best = b; best_sim = sim
                if best:
                    if best_sim < sim_min:
                        quarantine.append({"a": a.id, "b": best.id, "sim": best_sim, "metric": best_metric, "iter": it, "retries": 0})
                        continue
                    if best_sim < sim_thresh:
                        pairs.append((a, best, best_metric, "low-sim"))
                    else:
                        pairs.append((a, best, best_metric, "high-sim"))
                    used.add(a.id); used.add(best.id)
            if candidate_list:
                candidate_list_sorted = sorted(candidate_list, key=lambda x: -x[5])[:50]
                log(f"Candidate summary (top 50) iter={it} total_candidates={len(candidate_list)}")
                for ca, cb, sim, sa, sb, m in candidate_list_sorted[:20]:
                    _append_debug(f"CAND {ca} {cb} sim={sim:.4f} sa={sa:.4f} sb={sb:.4f} metric={m:.6f}")
            else:
                log("No candidate pairs generated this iteration")
            if not pairs:
                log("No pairs selected this iteration; will attempt quarantine retry or relax-run later")
                break
            for a_node, b_node, metric, tag in pairs:
                try:
                    vecs = [v for v in (a_node.vec, b_node.vec) if v is not None]
                    if not vecs:
                        continue
                    seed_vec = mean_vec(vecs)
                    shards = sorted(set(a_node.shards + b_node.shards))
                    diffs = []; ids = []
                    for n in (a_node, b_node):
                        if n.vec is not None:
                            d = vec_sub(n.vec, seed_vec)
                            if d is not None:
                                diffs.append(d); ids.append(n.shards[0] if n.shards else n.id)
                    qvecs, meta = quantize_list(diffs, bits=quant_bits) if diffs else ([], {})
                    diffs_map = {ids[i]: qvecs[i] for i in range(len(ids))} if qvecs else {}
                    seed = SeedNode(id=uid("seed-"), seed_vec=seed_vec, members=shards, diffs=diffs_map, quant_meta=meta)
                    if tag == "low-sim":
                        seed.quant_meta["low_sim_flag"] = True
                        seed.quant_meta["orig_metric"] = metric
                    self.seed_index.register(seed)
                    with self.registry.lock:
                        self.registry.nodes.pop(a_node.id, None)
                        self.registry.nodes.pop(b_node.id, None)
                        merged_ent = EntNode(id=uid("ent-"), vec=seed_vec, shards=shards, score=(a_node.score + b_node.score))
                        self.registry.nodes[merged_ent.id] = merged_ent
                        try:
                            safe_write_json(self.registry.path, {"nodes": {nid:n.to_dict() for nid,n in self.registry.nodes.items()}})
                        except Exception:
                            pass
                    Ledger.record("1PLUSNEG1_MERGE", seed.id, {"from":[a_node.id, b_node.id], "shards": len(shards), "metric": metric, "tag": tag})
                    append_comp_log({"ts": now_ts(), "seed": seed.id, "from":[a_node.id, b_node.id], "members": len(shards), "type":"1+-1", "metric": metric, "tag": tag})
                    log(f"Merged {a_node.id}+{b_node.id} -> {seed.id} tag={tag} metric={metric:.6f}")
                    merged_total += 1
                except Exception:
                    log_exc("merge error")
            try:
                self.registry.save(); self.seed_index.save(); Ledger.dump()
            except Exception:
                log_exc("post-merge persist error")
        # dedupe and save quarantine
        if quarantine:
            uniq = {}
            for q in quarantine:
                key = f"{q['a']}::{q['b']}"
                if key not in uniq:
                    uniq[key] = q
                else:
                    uniq[key]["retries"] = uniq[key].get("retries",0) + 1
            save_quarantine(list(uniq.values()))
            log(f"Quarantine saved {len(uniq)} pairs for retry")
        return {"merged": merged_total, "iters": it, "remaining": len(self.registry.nodes)}

    # Method B: force cluster and merge (aggressive)
    def force_cluster_and_merge(self, eps:float=0.25, min_members:int=2, quant_bits:int=cfg.quant_bits):
        nodes = list(self.registry.nodes.values())
        if len(nodes) < 2:
            log("force_cluster: not enough nodes")
            return {"forced":0}
        clusters: List[List[EntNode]] = []
        used = set()
        for i, a in enumerate(nodes):
            if a.id in used:
                continue
            cluster = [a]; used.add(a.id)
            for b in nodes[i+1:]:
                if b.id in used:
                    continue
                try:
                    if _cosine(a.vec, b.vec) >= eps:
                        cluster.append(b); used.add(b.id)
                except Exception:
                    continue
            if len(cluster) >= min_members:
                clusters.append(cluster)
        forced = 0
        for cl in clusters:
            try:
                vecs = [n.vec for n in cl if n.vec is not None]
                if not vecs:
                    continue
                seed_vec = mean_vec(vecs)
                shards = []
                diffs = []; ids = []
                for n in cl:
                    shards.extend(n.shards)
                    if n.vec is not None:
                        d = vec_sub(n.vec, seed_vec)
                        if d is not None:
                            diffs.append(d); ids.append(n.shards[0] if n.shards else n.id)
                qvecs, meta = quantize_list(diffs, bits=quant_bits) if diffs else ([], {})
                diffs_map = {ids[i]: qvecs[i] for i in range(len(ids))} if qvecs else {}
                seed = SeedNode(id=uid("seed-"), seed_vec=seed_vec, members=sorted(set(shards)), diffs=diffs_map, quant_meta=meta)
                seed.quant_meta["force_cluster_flag"] = True
                self.seed_index.register(seed)
                with self.registry.lock:
                    for n in cl:
                        self.registry.nodes.pop(n.id, None)
                    merged_ent = EntNode(id=uid("ent-"), vec=seed_vec, shards=sorted(set(shards)), score=sum(n.score for n in cl))
                    self.registry.nodes[merged_ent.id] = merged_ent
                    try:
                        safe_write_json(self.registry.path, {"nodes": {nid:n.to_dict() for nid,n in self.registry.nodes.items()}})
                    except Exception:
                        pass
                Ledger.record("FORCE_CLUSTER_MERGE", seed.id, {"from":[n.id for n in cl], "members": len(shards), "eps": eps})
                append_comp_log({"ts": now_ts(), "seed": seed.id, "from":[n.id for n in cl], "members": len(shards), "type":"force-cluster", "eps": eps})
                log(f"Force-cluster created seed {seed.id} from {[n.id for n in cl]} eps={eps}")
                forced += 1
            except Exception:
                log_exc("force-cluster merge error")
        try:
            self.registry.save(); self.seed_index.save(); Ledger.dump()
        except Exception:
            log_exc("force-cluster persist error")
        return {"forced": forced}

# -------------------------
# LazyExpander (unchanged)
# -------------------------
class LazyExpander:
    def __init__(self, seed_index:SeedIndex):
        self.seed_index = seed_index
        self.cache = {}
        self.lock = threading.RLock()

    def quick_holo(self, seed:SeedNode, query_vec:Optional[List[float]]=None, alpha:float=0.6):
        if query_vec is None:
            emb = seed.seed_vec
        else:
            emb = vec_add([x*alpha for x in query_vec], [x*(1-alpha) for x in seed.seed_vec]) if seed.seed_vec else query_vec
        delta = _cosine(seed.seed_vec, emb) if seed.seed_vec else 0.0
        holo = {"id": uid("h-"), "embedding": emb, "confidence": 0.75, "seed": seed.id, "delta": delta}
        Ledger.record("SEED_QUICK", holo["id"], {"seed":seed.id})
        return holo

    def expand(self, seed:SeedNode, top_n:int=6):
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
                        members.append({"id":mid,"embedding":emb})
                    except Exception:
                        members.append({"id":mid})
                else:
                    members.append({"id":mid})
        else:
            for mid in seed.members[:top_n]:
                members.append({"id":mid})
        res = {"seed":seed.id,"members":members,"ts":time.time()}
        with self.lock:
            self.cache[seed.id] = res
        Ledger.record("SEED_EXPAND", seed.id, {"members":len(members)})
        return res

# -------------------------
# Self-bootstrap: ingest shards or inject samples
# -------------------------
def _ingest_from_shards(registry:EntRegistry, max_import:int=1024):
    if not os.path.isdir(cfg.shard_dir):
        return 0
    files = sorted(os.listdir(cfg.shard_dir))
    imported = 0
    seen_hashes = set()
    for fname in files[:max_import]:
        fpath = os.path.join(cfg.shard_dir, fname)
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
        for i in range(cfg.dim):
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

_injected_counter_lock = threading.RLock()
_injected_counter_path = os.path.join(cfg.data_dir, "injected_count.json")

def _get_injected_count():
    try:
        if os.path.exists(_injected_counter_path):
            with open(_injected_counter_path, "r", encoding="utf-8") as f:
                return int(json.load(f).get("count", 0))
    except Exception:
        pass
    return 0

def _set_injected_count(n:int):
    try:
        safe_write_json(_injected_counter_path, {"count": n})
    except Exception:
        pass

def _inject_animation_samples(registry:EntRegistry, count:int = 24):
    samples = [
        "pocket infinite storage", "memory bread copy restore", "memory camera snapshot replay",
        "time cloth restore state", "memory disk compress replay", "memory capsule compress small",
        "holographic pocket seed aggregator", "seed singularity compressed origin", "lazy expansion reconstruct"
    ]
    injected = 0
    i = 0
    count = min(count, cfg.max_auto_inject)
    with _injected_counter_lock:
        already = _get_injected_count()
        to_inject = max(0, count - already)
        while injected < to_inject:
            s = samples[i % len(samples)] + f" sample-{already+injected}"
            import hashlib
            h = hashlib.sha256(s.encode('utf-8')).digest()
            vec = []
            for k in range(cfg.dim):
                b = h[k % len(h)]
                val = ((b / 255.0) * 0.6) - 0.3
                vec.append(val)
            nid = uid("ent-")
            registry.register(EntNode(id=nid, vec=vec, shards=[f"anim-{already+injected}"]))
            injected += 1
            i += 1
        if injected > 0:
            _set_injected_count(already + injected)
    return injected

# -------------------------
# Backup & rollback helpers
# -------------------------
def _prune_backups():
    try:
        items = sorted(os.listdir(cfg.backup_dir))
        if len(items) <= cfg.max_backup_keep:
            return
        for old in items[:-cfg.max_backup_keep]:
            p = os.path.join(cfg.backup_dir, old)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
            except Exception:
                pass
    except Exception:
        pass

def _backup_state():
    try:
        ts = int(time.time())
        dest = os.path.join(cfg.backup_dir, f"backup_{ts}")
        os.makedirs(dest, exist_ok=True)
        for p in (cfg.ent_path, cfg.seed_path, cfg.ledger_path, cfg.comp_log):
            if os.path.exists(p):
                try:
                    shutil.copy2(p, dest)
                except Exception:
                    pass
        log(f"Backup created at {dest}")
        _prune_backups()
        return dest
    except Exception as e:
        log(f"Backup failed: {e}")
        return None

def rollback_from_backup(backup_dir: str):
    try:
        for fname in ("ents.json","seeds.json","ledger.json","compression_log.json"):
            src = os.path.join(backup_dir, fname)
            dst = os.path.join(cfg.data_dir, fname)
            if os.path.exists(src):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass
        log(f"Rollback applied from {backup_dir}")
    except Exception as e:
        log(f"Rollback failed: {e}")

# -------------------------
# Quarantine retry & sublimation (embedded)
# -------------------------
def retry_quarantine_and_sublimate(compressor: Compressor, top_k:int=20, relax_steps:List[float]=[0.55,0.50], interp_steps:int=5, perturb_sigma:float=0.02, cluster_merge:bool=False):
    """
    Retry and 'sublimate' pairs from quarantine.
    Parameters:
      compressor: Compressor instance
      top_k: process top_k pairs by metric
      relax_steps: sequence of sim thresholds to try
      interp_steps: number of interpolation steps between vectors (inclusive)
      perturb_sigma: gaussian noise stddev for augmentation
      cluster_merge: if True, cluster related quarantine pairs first (not implemented here)
    """
    qpath = cfg.quarantine_path
    if not os.path.exists(qpath):
        log("retry_quarantine: no quarantine file")
        return {"tried":0,"merged":0}
    try:
        q = json.load(open(qpath, "r", encoding="utf-8"))
    except Exception:
        log("retry_quarantine: failed to load quarantine")
        return {"tried":0,"merged":0}
    q_sorted = sorted(q, key=lambda x: -float(x.get("metric",0)))[:top_k]
    tried = 0; merged = 0; new_quarantine = []

    def attempt_merge_by_vec(a_id, b_id, cand_vec, sim_target):
        nonlocal merged
        reg = compressor.registry
        a = reg.nodes.get(a_id); b = reg.nodes.get(b_id)
        if not a or not b:
            return False
        sim_a = _cosine(a.vec, cand_vec)
        sim_b = _cosine(b.vec, cand_vec)
        if sim_a >= sim_target and sim_b >= sim_target:
            shards = sorted(set(a.shards + b.shards))
            diffs=[]; ids=[]
            for n in (a,b):
                if n.vec is not None:
                    d = vec_sub(n.vec, cand_vec)
                    if d is not None:
                        diffs.append(d); ids.append(n.shards[0] if n.shards else n.id)
            qvecs, meta = quantize_list(diffs, bits=cfg.quant_bits) if diffs else ([],{})
            diffs_map = {ids[i]: qvecs[i] for i in range(len(ids))} if qvecs else {}
            seed = SeedNode(id=uid("seed-"), seed_vec=cand_vec, members=shards, diffs=diffs_map, quant_meta=meta)
            seed.quant_meta["sublimated_flag"] = True
            compressor.seed_index.register(seed)
            with compressor.registry.lock:
                compressor.registry.nodes.pop(a.id, None)
                compressor.registry.nodes.pop(b.id, None)
                merged_ent = EntNode(id=uid("ent-"), vec=cand_vec, shards=shards, score=(a.score + b.score))
                compressor.registry.nodes[merged_ent.id] = merged_ent
                try:
                    safe_write_json(compressor.registry.path, {"nodes": {nid:n.to_dict() for nid,n in compressor.registry.nodes.items()}})
                except Exception:
                    pass
            Ledger.record("SUBLIMATE_MERGE", seed.id, {"from":[a.id,b.id],"sim_a":sim_a,"sim_b":sim_b,"target":sim_target})
            append_comp_log({"ts": now_ts(), "seed": seed.id, "from":[a.id,b.id], "members": len(shards), "type":"sublimate", "sim_a":sim_a, "sim_b":sim_b, "target":sim_target})
            log(f"Sublimated merge {a.id}+{b.id} -> {seed.id} sim_a={sim_a:.4f} sim_b={sim_b:.4f} target={sim_target:.3f}")
            merged += 1
            return True
        return False

    for item in q_sorted:
        tried += 1
        a = item.get("a"); b = item.get("b")
        base_metric = float(item.get("metric",0))
        reg = compressor.registry
        na = reg.nodes.get(a); nb = reg.nodes.get(b)
        if not na or not nb:
            continue
        vecs = []
        for t in range(interp_steps+1):
            alpha = t / max(1, interp_steps)
            cand = [(1-alpha)*x + alpha*y for x,y in zip(na.vec, nb.vec)]
            vecs.append(cand)
            for p in range(2):
                pert = [v + random.gauss(0, perturb_sigma) for v in cand]
                vecs.append(pert)
        merged_flag = False
        for sim_target in relax_steps:
            if merged_flag: break
            for cand in vecs:
                if attempt_merge_by_vec(a, b, cand, sim_target):
                    merged_flag = True
                    break
        if not merged_flag:
            item["retries"] = item.get("retries",0) + 1
            if item["retries"] < cfg.quarantine_retry_limit:
                new_quarantine.append(item)
            else:
                item["exhausted"] = True
                new_quarantine.append(item)
    try:
        safe_write_json(cfg.quarantine_path, new_quarantine)
    except Exception:
        pass
    try:
        compressor.registry.save(); compressor.seed_index.save(); Ledger.dump()
    except Exception:
        pass
    return {"tried":tried,"merged":merged}

# -------------------------
# Autonomous pipeline driver & watcher
# -------------------------
class AutoDriver:
    def __init__(self):
        self.registry = EntRegistry()
        self.seed_index = SeedIndex()
        self.comp = Compressor(self.registry, self.seed_index)
        self.lock = threading.RLock()
        self.running = False
        self.sim = cfg.default_sim
        self.iters = cfg.default_iters
        self.min_group = cfg.min_group
        self.last_activity = time.time()
        self.metrics = {"runs":0,"merged":0,"seeds":0,"injected":_get_injected_count()}
        self._shutdown_requested = False

    def request_shutdown(self):
        self._shutdown_requested = True

    def ensure_data(self):
        if not self.registry.nodes:
            imported = _ingest_from_shards(self.registry, max_import=1024)
            if imported:
                log(f"AutoDriver: imported {imported} shards")
            else:
                injected = _inject_animation_samples(self.registry, count=cfg.max_auto_inject)
                if injected:
                    self.metrics["injected"] += injected
                log(f"AutoDriver: injected {injected} samples")

    def run_once(self, relax_sim: Optional[float] = None, force_cluster: bool = False, cluster_eps: float = 0.25):
        with self.lock:
            if self._shutdown_requested:
                log("AutoDriver: shutdown requested; skipping run")
                return
            self.ensure_data()
            if not self.registry.nodes:
                log("AutoDriver: no nodes to process")
                return
            backup = _backup_state()
            try:
                sim_to_use = relax_sim if relax_sim is not None else self.sim
                log(f"AutoDriver: running fusion with sim={sim_to_use:.3f} nodes={len(self.registry.nodes)}")
                res1 = self.comp.complementary_sublimate_flexible(sim_thresh=sim_to_use, sim_min=cfg.sim_min, max_iters=2)
                res2 = self.comp.build_seeds(sim_thresh=sim_to_use, min_group=self.min_group, quant_bits=cfg.quant_bits)
                merged = res1.get("merged",0) + res2.get("created",0)
                forced = 0
                # If quarantine exists and no merges, attempt automatic quarantine retry/sublimation
                try:
                    if merged == 0 and os.path.exists(cfg.quarantine_path):
                        log("AutoDriver: attempting automatic quarantine retry/sublimation")
                        retry_res = retry_quarantine_and_sublimate(self.comp, top_k=60, relax_steps=[0.55,0.50,0.45], interp_steps=6, perturb_sigma=0.03)
                        log(f"AutoDriver: quarantine retry result: {retry_res}")
                        merged += retry_res.get("merged",0)
                except Exception:
                    log_exc("auto quarantine retry error")
                if merged == 0 and force_cluster:
                    log("AutoDriver: no merges from flexible path; running force-cluster as backup")
                    forced_res = self.comp.force_cluster_and_merge(eps=cluster_eps, min_members=2, quant_bits=cfg.quant_bits)
                    forced = forced_res.get("forced", 0)
                self.metrics["runs"] += 1
                self.metrics["merged"] += merged + forced
                self.metrics["seeds"] += res2.get("created",0)
                self.last_activity = time.time()
                log(f"AutoDriver: run completed merged={merged} forced={forced} seeds_created={res2.get('created',0)} remaining_nodes={len(self.registry.nodes)}")
                if merged + forced > 0:
                    self.sim = min(0.995, self.sim + 0.01)
                else:
                    self.sim = max(0.50, self.sim - 0.02)
                self.registry.save(); self.seed_index.save(); Ledger.dump()
            except Exception as e:
                log(f"AutoDriver: error during run: {e}")
                log_exc("run_once exception")
                if backup:
                    rollback_from_backup(backup)

    def watch_loop(self, poll_interval: float = cfg.poll_interval):
        self.running = True
        log("AutoDriver: entering watch loop")
        self.run_once()
        while self.running and not self._shutdown_requested:
            if os.path.exists(cfg.shutdown_signal):
                log("AutoDriver: shutdown signal detected; exiting watch loop")
                break
            try:
                new_found = False
                for fname in os.listdir(cfg.shard_dir):
                    fpath = os.path.join(cfg.shard_dir, fname)
                    try:
                        mtime = os.path.getmtime(fpath)
                        if mtime > self.last_activity - 1:
                            new_found = True
                            break
                    except Exception:
                        continue
                if new_found:
                    log("AutoDriver: new shard detected -> triggering run")
                    self.run_once()
                if time.time() - self.last_activity > cfg.idle_run_seconds:
                    log("AutoDriver: idle timeout -> periodic run")
                    self.run_once()
            except Exception as e:
                log(f"AutoDriver: watch loop error: {e}")
                log_exc("watch_loop exception")
            time.sleep(poll_interval)
        self.running = False
        log("AutoDriver: watch loop ended")

# -------------------------
# Optional minimal HTTP status server
# -------------------------
class StatusHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/status"):
            try:
                body = {
                    "ts": now_ts(),
                    "ents": len(EntRegistry().nodes),
                    "seeds": len(SeedIndex().seeds),
                    "metrics": auto_driver.metrics if 'auto_driver' in globals() else {}
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode("utf-8"))
            except Exception:
                self.send_response(500); self.end_headers()
        elif self.path.startswith("/metrics"):
            try:
                m = auto_driver.metrics if 'auto_driver' in globals() else {}
                lines = []
                for k,v in m.items():
                    lines.append(f"qem_{k} {v}")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.end_headers()
                self.wfile.write("\n".join(lines).encode("utf-8"))
            except Exception:
                self.send_response(500); self.end_headers()
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, format, *args):
        return

def start_status_server(port:int):
    if port <= 0:
        return None
    try:
        server = socketserver.ThreadingTCPServer(("0.0.0.0", port), StatusHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        log(f"Status server listening on port {port}")
        return server
    except Exception as e:
        log(f"Failed to start status server: {e}")
        return None

# -------------------------
# Entrypoint
# -------------------------
def main(argv):
    parser = argparse.ArgumentParser(description="QEM Autonomous Final Embedded")
    parser.add_argument("--debug-run-once", action="store_true", help="Run one detailed debug iteration then exit")
    parser.add_argument("--no-auto", action="store_true", help="Do not auto-run (exit)")
    parser.add_argument("--relax-sim", type=float, default=None, help="Temporarily relax sim threshold for debug run")
    parser.add_argument("--force-cluster", action="store_true", help="If no merges, run force-cluster as backup")
    parser.add_argument("--cluster-eps", type=float, default=0.25, help="Eps for force-cluster")
    args = parser.parse_args(argv)

    global auto_driver
    auto_driver = AutoDriver()
    server = start_status_server(cfg.status_port)

    if args.no_auto:
        log("Auto-run disabled by --no-auto; exiting")
        return

    if args.debug_run_once:
        log("DEBUG RUN ONCE: starting detailed single run")
        try:
            with open(cfg.debug_log, "a", encoding="utf-8") as f:
                f.write(f"\n=== DEBUG RUN START {now_ts()} ===\n")
        except Exception:
            pass
        auto_driver.run_once(relax_sim=args.relax_sim, force_cluster=args.force_cluster, cluster_eps=args.cluster_eps)
        log("DEBUG RUN ONCE: finished; check qem_cloud_data/debug_log.txt for details")
        if server:
            try:
                server.shutdown(); server.server_close()
            except Exception:
                pass
        return

    try:
        auto_driver.watch_loop(poll_interval=cfg.poll_interval)
    except KeyboardInterrupt:
        log("KeyboardInterrupt received; shutting down")
    finally:
        if server:
            try:
                server.shutdown(); server.server_close()
            except Exception:
                pass
        Ledger.dump()
        log("Autonomous process exiting")

if __name__ == "__main__":
    main(sys.argv[1:])
