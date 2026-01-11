# fused_evolution_mobile_fixed.py
# Mobile-friendly fused evolution (fixed heap comparison bug)
# Save and run with Python 3.8+ on Termux / Pydroid: python3 fused_evolution_mobile_fixed.py

import os, sys, time, json, math, random, uuid, signal, gc, heapq
from copy import deepcopy

# -------------------------
# Mobile CONFIG
# -------------------------
CONFIG = {
    "OUTPUT_DIR": "fused_mobile_output",
    "SEED": 1234,
    "GENERATIONS": 6,
    "POP_MATERIALS": 6,
    "POP_OPERATORS": 6,
    "CANDIDATES_PER_GEN": 20,
    "APPLY_PER_MATERIAL": 2,
    "ELITE_MATERIALS": 2,
    "ELITE_OPERATORS": 2,
    "MUTATION_RATE": 0.18,
    "CROSSOVER_RATE": 0.5,
    "SAVE_INTERVAL": 2,
    "MAX_GEN_TIME": 10,
    "TOP_K_MATERIALS": 40,
    "D_WEIGHT": 1.5,
    "SUM_WEIGHT": 0.2,
    "BALANCE_PENALTY": 0.35,
    "COUPLING_TAG": "couple",
    "COUPLING_BOOST": 0.06,
    "SATURATION_PENALTY": 0.6,
    "SIGTERM_SAVE": True
}

# -------------------------
# Base materials
# -------------------------
BASE_MATERIALS = {
    "Al": {"manifest": 0.45, "latent": 0.02, "d": 0.3},
    "Borosilicate": {"manifest": 0.30, "latent": 0.01, "d": 0.3},
    "Ti_selfheal": {"manifest": 0.85, "latent": 0.30, "d": 0.3},
    "Graphene_Al2O3": {"manifest": 0.70, "latent": 0.25, "d": 0.3},
}

# -------------------------
# Lightweight Operator
# -------------------------
class Operator:
    __slots__ = ("id","params","tags","strength","age","origin")
    def __init__(self, params=None, tags=None, strength=1.0, origin="seed"):
        self.id = str(uuid.uuid4())[:8]
        self.params = dict(params or {})
        self.tags = set(tags or [])
        self.strength = float(strength)
        self.age = 0
        self.origin = origin
    def copy(self):
        op = Operator(params=self.params.copy(), tags=list(self.tags), strength=self.strength, origin=self.origin)
        op.id = self.id
        op.age = self.age
        return op
    def to_dict(self):
        return {"id":self.id,"params":self.params,"tags":list(self.tags),"strength":self.strength,"age":self.age,"origin":self.origin}

# -------------------------
# Utilities
# -------------------------
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
random.seed(CONFIG["SEED"])

def safe_write(path, obj):
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        print("Write failed:", e)

def sigterm_handler(signum, frame):
    print("SIGTERM received, saving snapshot")
    if CONFIG["SIGTERM_SAVE"]:
        safe_write(os.path.join(CONFIG["OUTPUT_DIR"], "sigterm_snapshot.json"), {"time":time.time()})
    sys.exit(1)

signal.signal(signal.SIGTERM, sigterm_handler)

# -------------------------
# Numeric collide and operator ops
# -------------------------
def numeric_collide(a,b,rng):
    if a is None and b is None: return 0.0
    if a is None: return float(b)
    if b is None: return float(a)
    aa,bb = float(a), float(b)
    if aa*bb < 0:
        return rng.gauss(0.0, max(abs(aa),abs(bb))*0.02 + 1e-6)
    avg = 0.5*(aa+bb)
    return avg + rng.gauss(0.0, abs(avg)*0.03 + 1e-6)

def collide_dict(pa,pb,rng):
    keys = set(pa.keys()) | set(pb.keys())
    out = {}
    for k in keys:
        va = pa.get(k)
        vb = pb.get(k)
        if isinstance(va,(int,float)) or isinstance(vb,(int,float)):
            out[k] = numeric_collide(va,vb,rng)
        else:
            if va == vb:
                out[k] = va
            else:
                if isinstance(va,str) and isinstance(vb,str):
                    out[k] = f"{va}__{vb}"
                else:
                    out[k] = va if va is not None else vb
    return out

def operator_collide(opA, opB, rng):
    new_params = collide_dict(opA.params, opB.params, rng)
    new_tags = set(opA.tags) | set(opB.tags)
    for a in opA.tags:
        for b in opB.tags:
            new_tags.add(f"{a}__{b}")
    base_strength = 0.5*(opA.strength+opB.strength)
    amp = 1.0 + (0.12 if any("__" in t for t in new_tags) else 0.0)
    new_strength = max(0.01, base_strength*amp + rng.gauss(0.0,0.05))
    new_op = Operator(params=new_params, tags=list(new_tags), strength=new_strength, origin="collide")
    if rng.random() < CONFIG["MUTATION_RATE"]:
        operator_mutate(new_op, rng)
    return new_op

def operator_mutate(op, rng):
    for k,v in list(op.params.items()):
        if isinstance(v,(int,float)):
            op.params[k] = float(v * (1.0 + rng.gauss(0.0,0.08)))
    if rng.random() < 0.25:
        op.tags.add("couple")
    if rng.random() < 0.12 and len(op.tags)>0:
        op.tags.pop()
    op.strength = max(0.01, op.strength*(1.0 + rng.gauss(0.0,0.05)))
    op.age = 0

# -------------------------
# Apply operator to material
# -------------------------
def apply_operator_to_material(material, operator, rng):
    manifest = float(material.get("manifest",0.0))
    latent = float(material.get("latent",0.0))
    d = float(material.get("d",0.3))
    p = operator.params
    inj = abs(p.get("injection_amp", p.get("inj",1e-4)))
    diff = abs(p.get("diffusion_nu", p.get("kappa",1e-3)))
    xi = float(p.get("xi", 0.02))
    lam = float(p.get("lambda", 0.12))
    influence_m = operator.strength * (0.5*inj + 0.5*(1.0/(1.0+diff)))
    influence_l = operator.strength * (0.5*xi + 0.5*lam)
    manifest = min(1.0, manifest + 0.10 * influence_m * (1.0 - manifest))
    latent = min(1.0, latent + 0.08 * influence_l * (1.0 - latent))
    closeness = 1.0 - abs(manifest - latent)
    tag_boost = CONFIG["COUPLING_BOOST"] * sum(1 for t in operator.tags if CONFIG["COUPLING_TAG"] in t)
    d = min(1.0, d + 0.06 * closeness + tag_boost)
    phi = rng.gauss(0.0, 0.01) + 0.001*(manifest - latent)
    return {"manifest":manifest,"latent":latent,"d":d,"phi":phi,"last_op":operator.id}

# -------------------------
# Evaluation
# -------------------------
def balance_operator(alpha,beta,gamma,psi_m,psi_l):
    return (alpha*psi_m + beta*psi_l)/max(gamma,1e-12)

def evaluate_material(material, alpha=0.4,beta=0.6,gamma=0.5):
    psi_m = material.get("manifest",0.0)
    psi_l = material.get("latent",0.0)
    d = material.get("d",0.0)
    psi_balance = balance_operator(alpha,beta,gamma,psi_m,psi_l)
    sum_term = (psi_m + psi_l)
    imbalance = abs(psi_m - psi_l)
    saturation = max(0.0, (psi_m + psi_l) - 1.6)
    score = (CONFIG["D_WEIGHT"] * d) + (CONFIG["SUM_WEIGHT"] * sum_term) - (CONFIG["BALANCE_PENALTY"] * imbalance) - (CONFIG["SATURATION_PENALTY"] * saturation) - abs(psi_balance)*0.2
    return {"psi_balance":psi_balance,"d":d,"score":float(score)}

# -------------------------
# Candidate generator
# -------------------------
def operator_candidate_generator(operator_pool, rng, n, crossover_rate):
    for _ in range(n):
        if rng.random() < crossover_rate and len(operator_pool) >= 2:
            a,b = rng.sample(operator_pool,2)
            yield operator_collide(a,b,rng)
        else:
            parent = rng.choice(operator_pool)
            op = parent.copy()
            op.id = str(uuid.uuid4())[:8]
            if rng.random() < CONFIG["MUTATION_RATE"]:
                operator_mutate(op,rng)
            yield op

# -------------------------
# Init pools
# -------------------------
def init_material_pool(pop, rng):
    names = list(BASE_MATERIALS.keys())
    pool = []
    for i in range(pop):
        base = deepcopy(BASE_MATERIALS[rng.choice(names)])
        base["manifest"] = float(max(0.0,min(1.0, base["manifest"] + rng.gauss(0.0,0.02))))
        base["latent"] = float(max(0.0,min(1.0, base["latent"] + rng.gauss(0.0,0.02))))
        base["d"] = float(max(0.0,min(1.0, base.get("d",0.3) + rng.gauss(0.0,0.03))))
        base["id"] = f"m{i}"
        pool.append(base)
    return pool

def init_operator_pool(pop, rng):
    pool = []
    base_ops = [
        Operator(params={"injection_amp":1e-4,"xi":0.02,"kappa":1e-3}, tags=["couple","inject"], strength=1.0),
        Operator(params={"injection_amp":2e-4,"xi":0.03,"kappa":2e-3}, tags=["stabilize","diffuse"], strength=0.9),
        Operator(params={"injection_amp":5e-5,"xi":0.01,"kappa":5e-4}, tags=["morph","surface"], strength=0.8),
    ]
    for i in range(pop):
        base = deepcopy(random.choice(base_ops))
        base.id = str(uuid.uuid4())[:8]
        base.origin = "init"
        operator_mutate(base, random)
        pool.append(base)
    return pool

# -------------------------
# Evolution loop (mobile, fixed)
# -------------------------
def evolution_loop(generations):
    rng = random.Random(CONFIG["SEED"])
    material_pool = init_material_pool(CONFIG["POP_MATERIALS"], rng)
    operator_pool = init_operator_pool(CONFIG["POP_OPERATORS"], rng)
    history = []
    best_score = -1e9
    best_material = None

    for gen in range(generations):
        gen_start = time.time()
        print(f"\n=== Generation {gen+1} ===")
        # timeout guard
        def _timeout(signum, frame):
            raise TimeoutError("Generation timeout")
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(CONFIG["MAX_GEN_TIME"])

        try:
            op_gen = operator_candidate_generator(operator_pool, rng, CONFIG["CANDIDATES_PER_GEN"], CONFIG["CROSSOVER_RATE"])
            top_heap = []  # will store (score, counter, material)
            heap_counter = 0
            total_evaluated = 0
            buffer = []
            buffer_size = max(4, CONFIG["CANDIDATES_PER_GEN"]//8)
            for _ in range(buffer_size):
                try:
                    buffer.append(next(op_gen))
                except StopIteration:
                    break

            for mat in material_pool:
                for _ in range(CONFIG["APPLY_PER_MATERIAL"]):
                    if not buffer:
                        try:
                            buffer.append(next(op_gen))
                        except StopIteration:
                            break
                    op = rng.choice(buffer)
                    new_mat = apply_operator_to_material(mat, op, rng)
                    eval_res = evaluate_material(new_mat)
                    score = eval_res["score"]
                    total_evaluated += 1
                    if len(top_heap) < CONFIG["TOP_K_MATERIALS"]:
                        heapq.heappush(top_heap, (score, heap_counter, new_mat))
                        heap_counter += 1
                    else:
                        if score > top_heap[0][0]:
                            heapq.heapreplace(top_heap, (score, heap_counter, new_mat))
                            heap_counter += 1
                    if rng.random() < 0.2:
                        try:
                            buffer.append(next(op_gen))
                        except StopIteration:
                            pass

            # convert heap to materials_with_scores safely
            materials_with_scores = [(item[2], item[0]) for item in top_heap]
            if not materials_with_scores:
                print("No materials evaluated, stopping")
                break

            # selection
            elites = sorted(materials_with_scores, key=lambda x: x[1], reverse=True)[:CONFIG["ELITE_MATERIALS"]]
            elites = sorted([m for m,s in elites], key=lambda m: m.get("d",0.0), reverse=True)[:CONFIG["ELITE_MATERIALS"]]
            selected = elites[:]
            while len(selected) < CONFIG["POP_MATERIALS"]:
                a,b = random.sample(materials_with_scores,2)
                chosen = a[0] if a[1] > b[1] else b[0]
                if chosen.get("d",0.0) < 0.6 and random.random() < 0.6:
                    alt = max(materials_with_scores, key=lambda x: x[0].get("d",0.0))[0]
                    selected.append(alt)
                else:
                    selected.append(chosen)

            for i,m in enumerate(selected):
                m.setdefault("id", f"mat_gen{gen}_{i}")
                m["age"] = m.get("age",0) + 1
            material_pool = selected

            # operator update (light)
            all_ops = operator_pool[:] + buffer[:min(len(buffer), CONFIG["POP_OPERATORS"])]
            operators_with_scores = []
            for op in all_ops:
                fitness = (CONFIG["COUPLING_BOOST"] * sum(1 for t in op.tags if CONFIG["COUPLING_TAG"] in t)) + (1.0 - abs(op.strength - 1.0)*0.1)
                operators_with_scores.append((op, fitness))
            elite_ops = sorted(operators_with_scores, key=lambda x: x[1], reverse=True)[:CONFIG["ELITE_OPERATORS"]]
            next_pool = [op for op,score in sorted(operators_with_scores, key=lambda x: x[1], reverse=True)[:max(0, CONFIG["POP_OPERATORS"] - len(elite_ops))]]
            for e,score in elite_ops:
                if e not in next_pool:
                    next_pool.append(e)
            while len(next_pool) < CONFIG["POP_OPERATORS"]:
                clone = deepcopy(random.choice(next_pool)) if next_pool else deepcopy(random.choice(all_ops))
                clone.id = str(uuid.uuid4())[:8]
                operator_mutate(clone, random)
                next_pool.append(clone)
            for op in next_pool:
                op.age = op.age + 1
            operator_pool = next_pool

            top_materials = sorted([(item[2], item[0]) for item in top_heap], key=lambda x: x[1], reverse=True)[:3]
            gen_log = {
                "generation": gen,
                "num_evaluated": total_evaluated,
                "top_materials": [{"id":m.get("id"), "score":s, "manifest":m.get("manifest"), "latent":m.get("latent"), "d":m.get("d")} for m,s in top_materials],
                "operator_pool_size": len(operator_pool),
                "material_pool_size": len(material_pool)
            }
            history.append(gen_log)
            if top_materials:
                top_score, top_mat = top_materials[0][1], top_materials[0][0]
                if top_score > best_score:
                    best_score = top_score
                    best_material = top_mat

            del buffer, top_heap, materials_with_scores
            gc.collect()
            duration = time.time() - gen_start
            print(f"Evaluated ~{total_evaluated} materials, Best score so far: {best_score:.4f}, duration {duration:.2f}s")
            signal.alarm(0)

            if gen % CONFIG["SAVE_INTERVAL"] == 0 or gen == generations-1:
                safe_write(os.path.join(CONFIG["OUTPUT_DIR"], f"materials_gen{gen}.json"), material_pool)
                safe_write(os.path.join(CONFIG["OUTPUT_DIR"], f"operators_gen{gen}.json"), [op.to_dict() for op in operator_pool])
                safe_write(os.path.join(CONFIG["OUTPUT_DIR"], f"generation_{gen}_summary.json"), gen_log)

        except TimeoutError:
            print("Generation timed out, saving partial state")
            safe_write(os.path.join(CONFIG["OUTPUT_DIR"], f"timeout_gen{gen}.json"), {"gen":gen,"time":time.time()})
            signal.alarm(0)
            break
        except Exception as e:
            safe_write(os.path.join(CONFIG["OUTPUT_DIR"], "fatal_error.txt"), {"error":str(e),"gen":gen,"time":time.time()})
            print("Fatal error:", e)
            signal.alarm(0)
            break

    safe_write(os.path.join(CONFIG["OUTPUT_DIR"], "final_materials.json"), material_pool)
    safe_write(os.path.join(CONFIG["OUTPUT_DIR"], "final_operators.json"), [op.to_dict() for op in operator_pool])
    safe_write(os.path.join(CONFIG["OUTPUT_DIR"], "evolution_history.json"), history)
    print("\nEvolution complete. Best score:", best_score)
    if best_material:
        print("Best material:", best_material)
    return history, best_material

# -------------------------
# Entrypoint
# -------------------------
def main():
    if len(sys.argv) > 1:
        try:
            CONFIG["GENERATIONS"] = int(sys.argv[1])
        except:
            pass
    print("融合升华 Mobile Fixed — 启动")
    print("Config:", {k:CONFIG[k] for k in ("GENERATIONS","POP_MATERIALS","POP_OPERATORS","CANDIDATES_PER_GEN")})
    history, best = evolution_loop(CONFIG["GENERATIONS"])
    return history

if __name__ == "__main__":
    main()