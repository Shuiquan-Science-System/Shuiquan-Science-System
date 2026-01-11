# strategic_pinnacle_qizheng.py
"""
Strategic Pinnacle — Fusion of Stability and Surprise (single-file)
Description:
 - Two-layer strategy (Zheng: PPO for long-term stability; Qi: meta-controller triggers controlled unpredictability)
 - Latent modes (conservative/balanced/aggressive/deceptive)
 - Temperature scheduling, vulnerability assessment, controlled randomness, action clipping, opponent pool (data-driven optional)
 - Replaceable DomainEnv (if you define DomainEnv in domain_model.py it will be auto-loaded)
 - Example run: python strategic_pinnacle_qizheng.py --episodes 100 --resources 60 --train
Dependencies: numpy (required); optional: torch, pulp, pandas
"""

import os, sys, math, time, random, json, argparse, logging, shutil
from collections import deque, namedtuple, defaultdict, Counter
from typing import List, Dict, Any
import numpy as np

# Optional libraries
try:
    import pulp
    HAS_PULP = True
except Exception:
    HAS_PULP = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# -------------------------
# Configuration
# -------------------------
CONFIG = {
    "state_max": 200,
    "belief_bins": 201,
    "mcts_depth": 3,
    "mcts_rollouts": 3,
    "policy_candidates": 6,
    "deception_cost_lambda": 0.6,
    "resource_resilience": 12,
    "simulate_steps": 40,
    "decision_time_budget": 0.6,
    "scenario_count": 8,
    "exploration_bonus": 0.02,
    "random_seed": 42,
    "ppo_epochs": 4,
    "ppo_clip": 0.2,
    "ppo_lr": 1e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "batch_size": 128,
    "log_dir": "logs",
    "model_dir": "models",
    "audit_threshold_risk": 0.25,
    "long_term_weight": 0.65,
    "short_term_weight": 0.35,
    "min_reserve": 12,
    "eps": 1e-12,
    "latent_modes": ["conservative","balanced","aggressive","deceptive"],
    "base_temp": 1.0,
    "temp_scale": 1.2,
    "entropy_coef": 0.01,
    "max_force_by_mode": {"conservative":25,"balanced":30,"aggressive":40,"deceptive":30},
    "opponent_data_csv": "opponent_data.csv",
    "domain_model_py": "domain_model.py",
}

random.seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])
if HAS_TORCH:
    torch.manual_seed(CONFIG["random_seed"])

os.makedirs(CONFIG["log_dir"], exist_ok=True)
os.makedirs(CONFIG["model_dir"], exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(os.path.join(CONFIG["log_dir"], "engine.log")), logging.StreamHandler(sys.stdout)])

# -------------------------
# Utilities
# -------------------------
def safe_clip_prob(p):
    return np.clip(p, CONFIG["eps"], 1.0)

def safe_log(x):
    return np.log(np.clip(x, CONFIG["eps"], None))

def normalize(v):
    v = np.array(v, dtype=float)
    s = v.sum()
    if s == 0 or np.isnan(s):
        return np.ones_like(v) / len(v)
    return v / s

def kl_divergence(p, q):
    p = safe_clip_prob(np.array(p, dtype=float))
    q = safe_clip_prob(np.array(q, dtype=float))
    return float(np.sum(p * np.log(p / q)))

def entropy_of(belief):
    p = safe_clip_prob(belief)
    return float(-np.sum(p * np.log(p)))

def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

# -------------------------
# Domain Environment (default) Replaceable
# -------------------------
class DomainEnv:
    def __init__(self, true_state=None, context=None):
        self.max_s = CONFIG["state_max"]
        self.true_state = {"strength": true_state if true_state is not None else random.randint(40,160),
                           "context": context if context is not None else random.choice(["normal","peak","disrupted"])}
        self.step_count = 0
        self.noise_sigma = 6.0
    def sample_initial_state(self):
        return dict(self.true_state)
    def generate_observation(self):
        noise = random.gauss(0, self.noise_sigma)
        obs_val = max(0, min(self.max_s, int(round(self.true_state["strength"] + noise))))
        return {"strength_est": obs_val, "context": self.true_state["context"]}
    def step(self, action):
        self.step_count += 1
        reward = 0.0
        if action is None:
            reward = -0.01
        else:
            t = action.get("type")
            if t == "invest":
                force = action.get("force",0)
                p_success = 1/(1+math.exp((self.true_state["strength"]-force)/20.0))
                if random.random() < p_success:
                    improvement = int(max(1, force * random.uniform(0.12,0.35)))
                    self.true_state["strength"] = max(0, self.true_state["strength"] - improvement)
                    reward = 1.0 + improvement*0.02
                else:
                    reward = -0.15 - force*0.008
            elif t == "signal":
                reward = 0.03 - action.get("cost",0)*0.01
            elif t == "recon":
                self.noise_sigma = max(1.0, self.noise_sigma*0.9)
                reward = 0.01
            elif t == "innovate":
                reward = 0.01 + CONFIG["exploration_bonus"]
            else:
                reward = -0.02
        done = (self.true_state["strength"]==0) or (self.step_count>=CONFIG["simulate_steps"])
        obs = self.generate_observation()
        return obs, float(reward), bool(done), {}

# Auto-load user DomainEnv from domain_model.py if present
if os.path.exists(CONFIG["domain_model_py"]):
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("domain_model", CONFIG["domain_model_py"])
        domain_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(domain_model)
        if hasattr(domain_model, "DomainEnv"):
            DomainEnv = domain_model.DomainEnv
            logging.info("Loaded user DomainEnv from domain_model.py")
    except Exception as e:
        logging.warning(f"Failed to load domain_model.py: {e}")

# -------------------------
# Belief helpers
# -------------------------
STATE_SPACE = np.arange(0, CONFIG["state_max"]+1)
def compute_likelihood(observation, state_space=STATE_SPACE, obs_sigma=6.0):
    obs_val = observation["strength_est"]
    diffs = (obs_val - state_space)/obs_sigma
    like = np.exp(-0.5*diffs*diffs)
    return normalize(like)
def bayes_update(prior, observation, state_space=STATE_SPACE, obs_sigma=6.0):
    like = compute_likelihood(observation, state_space, obs_sigma)
    post = normalize(prior * like)
    return post

# -------------------------
# InfoAction / Deception primitives
# -------------------------
class InfoAction:
    def __init__(self, id, shift, cost, desc, risk=0.0):
        self.id = id; self.shift = shift; self.cost = cost; self.desc = desc; self.risk = risk
    def as_dict(self):
        return {"id":self.id,"shift":self.shift,"cost":self.cost,"desc":self.desc,"risk":self.risk}

def simulate_enemy_update(enemy_prior, info_action):
    shifted = np.roll(enemy_prior, info_action.shift)
    return normalize(shifted)

def compute_mislead_gain(predicted_enemy_b, true_belief):
    return kl_divergence(true_belief, predicted_enemy_b)

# -------------------------
# Opponent models and data-driven replacement
# -------------------------
class OpponentModel:
    def __init__(self, bias_shift=0, noise=0.0, counter_strength=0.1, policy=None):
        self.bias_shift = bias_shift
        self.noise = noise
        self.counter_strength = counter_strength
        self.policy = policy
    def update_belief(self, prior):
        b = np.roll(prior, self.bias_shift)
        if self.noise > 0:
            b = normalize(b + np.random.normal(0, self.noise, size=b.shape))
        return normalize(b)
    def respond(self, observation):
        if self.policy is not None:
            try:
                return self.policy(observation)
            except Exception:
                pass
        return {"type":"invest", "force": random.randint(5,30)}
    def estimate_counter_prob(self, info_action):
        base = min(1.0, abs(info_action.shift) / 12.0 + info_action.cost / 20.0)
        return min(1.0, base * (0.5 + self.counter_strength))

def build_opponent_pool():
    pool = [OpponentModel(bias_shift=0, noise=0.0, counter_strength=0.05),
            OpponentModel(bias_shift=3, noise=0.02, counter_strength=0.1),
            OpponentModel(bias_shift=-4, noise=0.03, counter_strength=0.12),
            OpponentModel(bias_shift=6, noise=0.05, counter_strength=0.2),
            OpponentModel(bias_shift=-2, noise=0.01, counter_strength=0.08)]
    return pool

OPPONENT_POOL = build_opponent_pool()

def load_opponent_data_and_build_policies(csv_path=CONFIG["opponent_data_csv"]):
    if not HAS_PANDAS or not os.path.exists(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path)
        policies = []
        grouped = df.groupby("opponent_id") if "opponent_id" in df.columns else [("default", df)]
        for name, g in grouped:
            mapping = {}
            for _, row in g.iterrows():
                obs = int(row.get("obs_strength", 0))
                key = int(round(obs/5.0))*5
                mapping.setdefault(key, []).append((row.get("action_type"), row.get("action_force",0), row.get("action_cost",0)))
            def make_policy(map_copy):
                def policy(obs):
                    s = int(obs.get("strength_est", 0))
                    key = int(round(s/5.0))*5
                    choices = map_copy.get(key, None)
                    if not choices:
                        return {"type":"invest", "force": random.randint(5,30)}
                    a = random.choice(choices)
                    if a[0] == "invest":
                        return {"type":"invest", "force": int(a[1])}
                    if a[0] == "signal":
                        return {"type":"signal", "cost": int(a[2])}
                    if a[0] == "recon":
                        return {"type":"recon"}
                    return {"type":"innovate"}
                return policy
            policies.append(make_policy(mapping))
        logging.info(f"Loaded {len(policies)} opponent policies from {csv_path}")
        return policies
    except Exception as e:
        logging.warning(f"Failed to load opponent data: {e}")
        return []

if HAS_PANDAS and os.path.exists(CONFIG["opponent_data_csv"]):
    opp_policies = load_opponent_data_and_build_policies(CONFIG["opponent_data_csv"])
    if opp_policies:
        for i, p in enumerate(opp_policies):
            if i < len(OPPONENT_POOL):
                base = OPPONENT_POOL[i]
                OPPONENT_POOL[i] = OpponentModel(bias_shift=base.bias_shift, noise=base.noise, counter_strength=base.counter_strength, policy=p)

# -------------------------
# Deception selection routine
# -------------------------
def deception_select(own_belief, enemy_model_prior, info_actions, cost_limit, max_risk=0.5):
    best=None; best_score=-1e9
    for a in info_actions:
        if a.risk > max_risk:
            continue
        gains = []
        for op in OPPONENT_POOL:
            enemy_prior = op.update_belief(enemy_model_prior)
            pred = simulate_enemy_update(enemy_prior, a)
            gain = compute_mislead_gain(pred, own_belief)
            counter_prob = op.estimate_counter_prob(a)
            net_gain = gain * (1 - counter_prob) - counter_prob * a.cost * 0.5
            gains.append(net_gain)
        avg_gain = float(np.mean(gains)) if gains else 0.0
        score = avg_gain - CONFIG["deception_cost_lambda"] * a.cost
        if a.cost <= cost_limit and score > best_score:
            best = a; best_score = score
    return best, best_score

# -------------------------
# Resource optimization
# -------------------------
def compute_expected_enemy(belief):
    return float(np.dot(STATE_SPACE, belief))

def resource_optimize(tasks, total_resources, belief, base_resilience=CONFIG["resource_resilience"]):
    expected_enemy = compute_expected_enemy(belief)
    resilience = int(base_resilience * (1.0 - expected_enemy / CONFIG["state_max"]))
    resilience = max(CONFIG["min_reserve"], resilience)
    dynamic_tasks = []
    for t in tasks:
        urgency = 1.0 + (expected_enemy / CONFIG["state_max"])
        dyn_value = t['value'] * urgency
        dynamic_tasks.append({"id": t['id'], "need": t['need'], "value": dyn_value})
    if HAS_PULP:
        try:
            prob = pulp.LpProblem("res_alloc", pulp.LpMaximize)
            x = {t['id']: pulp.LpVariable(f"x_{t['id']}", lowBound=0, upBound=t['need'], cat='Integer') for t in dynamic_tasks}
            prob += pulp.lpSum([t['value'] * x[t['id']] for t in dynamic_tasks]) + 0.01*(total_resources - pulp.lpSum([x[t['id']] for t in dynamic_tasks]))
            prob += pulp.lpSum([x[t['id']] for t in dynamic_tasks]) <= max(0, total_resources - resilience)
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            alloc = {t['id']: int(pulp.value(x[t['id']])) for t in dynamic_tasks}
            alloc['reserve'] = max(0, total_resources - sum(alloc.values()))
            return alloc
        except Exception:
            pass
    tasks_sorted = sorted(dynamic_tasks, key=lambda x: -x['value'])
    alloc = {}
    rem = max(0, total_resources - resilience)
    for t in tasks_sorted:
        a = min(t['need'], rem)
        alloc[t['id']] = a
        rem -= a
        if rem <= 0:
            break
    for t in dynamic_tasks:
        if t['id'] not in alloc:
            alloc[t['id']] = 0
    alloc['reserve'] = max(0, total_resources - sum([alloc[k] for k in alloc if k != 'reserve']))
    return alloc

# -------------------------
# Action mapping (reduced step sizes)
# -------------------------
ACTION_DIM = 12
def action_from_index(idx):
    # invest forces scaled by 3 to reduce single-step aggressiveness
    if idx < 8: return {"type":"invest","force":5+idx*3}
    elif idx == 8: return {"type":"signal","cost":5}
    elif idx == 9: return {"type":"recon"}
    else: return {"type":"innovate"}

# -------------------------
# PPO Agent (temperature-aware)
# -------------------------
if HAS_TORCH:
    class ActorCritic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(state_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
            self.actor = nn.Linear(256, action_dim)
            self.critic = nn.Linear(256, 1)
        def forward(self, x):
            h = self.fc(x)
            return self.actor(h), self.critic(h)

    class PPOAgent:
        def __init__(self, state_dim, action_dim, lr=CONFIG["ppo_lr"]):
            self.net = ActorCritic(state_dim, action_dim)
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
            self.action_dim = action_dim
        def act(self, state, temp=1.0):
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = self.net(s)
            logits = logits.detach().numpy()[0]
            # apply temperature
            scaled = logits / max(1e-6, temp)
            probs = np.exp(scaled - np.max(scaled))
            probs = probs / probs.sum()
            probs = np.clip(probs, CONFIG["eps"], 1.0)
            probs = probs / probs.sum()
            a = np.random.choice(self.action_dim, p=probs)
            return int(a), float(probs[a]), float(value.item())
        def evaluate(self, states, actions):
            s = torch.tensor(states, dtype=torch.float32)
            logits, values = self.net(s)
            log_probs = torch.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, torch.tensor(actions).unsqueeze(1)).squeeze(1)
            return action_log_probs, values.squeeze(1)
        def update(self, trajectories):
            states = np.vstack([t['states'] for t in trajectories if t['states'].shape[0]>0])
            actions = np.hstack([t['actions'] for t in trajectories if t['actions'].shape[0]>0])
            returns = np.hstack([t['returns'] for t in trajectories if t['returns'].shape[0]>0])
            advantages = np.hstack([t['advantages'] for t in trajectories if t['advantages'].shape[0]>0])
            if len(states)==0:
                return
            states_t = torch.tensor(states, dtype=torch.float32)
            actions_t = torch.tensor(actions, dtype=torch.long)
            returns_t = torch.tensor(returns, dtype=torch.float32)
            adv_t = torch.tensor(advantages, dtype=torch.float32)
            old_log_probs, _ = self.evaluate(states, actions)
            old_log_probs = old_log_probs.detach()
            dataset = torch.utils.data.TensorDataset(states_t, actions_t, returns_t, adv_t, old_log_probs)
            loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
            for _ in range(CONFIG["ppo_epochs"]):
                for b_states, b_actions, b_returns, b_adv, b_old_log in loader:
                    logits, values = self.net(b_states)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    action_log_probs = log_probs.gather(1, b_actions.unsqueeze(1)).squeeze(1)
                    ratio = torch.exp(action_log_probs - b_old_log)
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1.0 - CONFIG["ppo_clip"], 1.0 + CONFIG["ppo_clip"]) * b_adv
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = (b_returns - values.squeeze(1)).pow(2).mean()
                    loss = actor_loss + 0.5 * critic_loss - CONFIG["entropy_coef"] * (-(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=1).mean())
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    self.optimizer.step()
else:
    class PPOAgent:
        def __init__(self, state_dim, action_dim, lr=None):
            self.state_dim = state_dim; self.action_dim = action_dim
        def act(self, state, temp=1.0):
            # non-torch fallback: uniform sampling with temperature effect via exponentiation
            probs = np.ones(self.action_dim) / self.action_dim
            probs = probs ** (1.0 / max(1e-6, temp))
            probs = probs / probs.sum()
            a = np.random.choice(self.action_dim, p=probs)
            return int(a), float(probs[a]), 0.0
        def update(self, trajectories): pass

# -------------------------
# Short-term and long-term evaluation
# -------------------------
def mcts_rollout(belief, action, simulator_cls, depth=3, rollouts=3):
    total=0.0
    for _ in range(rollouts):
        sim = simulator_cls()
        obs, r, done, _ = sim.step(action)
        total += r
        for d in range(depth-1):
            if done: break
            a = random.choice([{"type":"invest","force":random.randint(5,30)},
                               {"type":"signal","cost":3},{"type":"recon"},{"type":"innovate"}])
            obs, r, done, _ = sim.step(a)
            total += r
    return total/rollouts

def system_projection_score(initial_state, policy_seq, steps=8):
    state = dict(initial_state)
    total_improvement = 0.0
    for a in policy_seq[:steps]:
        before = state["strength"]
        if a.get("type") == "invest":
            state["strength"] = max(0, state["strength"] - a.get("force",0)*0.12)
        elif a.get("type") == "innovate":
            state["strength"] = max(0, state["strength"] - 0.6)
        state["strength"] += 0.05*random.uniform(-1,1)
        after = state["strength"]
        total_improvement += (before - after)
    return float(total_improvement) / (steps + 1.0)

# -------------------------
# Policy distillation to simple rules
# -------------------------
def distill_policy_to_rules(policy_agent, sample_states=400):
    rules = []
    for _ in range(sample_states):
        s = np.random.randint(0, CONFIG["state_max"])
        belief = normalize(np.exp(-0.5 * ((np.arange(CONFIG["belief_bins"]) - s)/6.0)**2))
        idx, prob, val = policy_agent.act(belief, temp=CONFIG["base_temp"])
        rules.append({"state_center": int(s), "action_index": int(idx)})
    grouped = defaultdict(list)
    for r in rules:
        grouped[r["action_index"]].append(r["state_center"])
    distilled = [{"action_index": k, "mean_center": int(np.mean(v)), "count": len(v)} for k,v in grouped.items()]
    return distilled

# -------------------------
# OODA loop wrapper / Meta-controller
# -------------------------
class OODA:
    def __init__(self, engine): self.engine = engine
    def loop_once(self, prior, observation, resources):
        belief = bayes_update(prior, observation)
        decision = self.engine.decide(prior, observation, resources)
        return decision

# -------------------------
# Engine integrating Qi and Zheng
# -------------------------
Experience = namedtuple("Experience", ["belief","action_index","reward","next_obs","done"])
class StrategicPinnacle:
    def __init__(self, simulator_cls=DomainEnv, use_ppo=HAS_TORCH):
        self.simulator_cls = simulator_cls
        self.prior = normalize(np.ones(len(STATE_SPACE)))
        self.enemy_model_prior = normalize(np.ones(len(STATE_SPACE)))
        self.info_actions = [
            InfoAction(1, shift=6, cost=8, desc="announce expansion", risk=0.1),
            InfoAction(2, shift=-4, cost=5, desc="leak supply issue", risk=0.3),
            InfoAction(3, shift=3, cost=3, desc="feint marketing", risk=0.05),
            InfoAction(4, shift=8, cost=12, desc="major PR push", risk=0.4)
        ]
        self.opponent_pool = OPPONENT_POOL
        self.policy = PPOAgent(state_dim=len(STATE_SPACE), action_dim=ACTION_DIM) if use_ppo else PPOAgent(state_dim=len(STATE_SPACE), action_dim=ACTION_DIM)
        self.metrics = {"episodes":0,"wins":0,"total_reward":0.0,"action_counter":Counter(),"deception_counter":Counter()}
        self.experience_buffer = deque(maxlen=200000)
        self.ooda = OODA(self)
        self.audit_log = []
        self.current_mode = "balanced"

    def compute_opponent_vulnerability(self, belief):
        # evaluate average KL shift of info actions across opponent pool
        scores = []
        for a in self.info_actions:
            gains = []
            for op in self.opponent_pool:
                enemy_prior = op.update_belief(self.enemy_model_prior)
                pred = simulate_enemy_update(enemy_prior, a)
                gain = compute_mislead_gain(pred, belief)
                gains.append(gain)
            scores.append(np.mean(gains))
        return max(scores) if scores else 0.0

    def decide(self, prior_belief, observation, resources, time_budget=CONFIG["decision_time_budget"]):
        start=time.time()
        belief = bayes_update(prior_belief, observation)
        be_entropy = entropy_of(belief)
        opponent_vul = self.compute_opponent_vulnerability(belief)
        # latent-mode influence: current_mode is set per episode externally
        z = getattr(self, "current_mode", "balanced")
        # dynamic temperature: higher when belief entropy is high -> more randomness (Qi)
        temp = CONFIG["base_temp"] * (1.0 + (be_entropy / math.log(len(belief)+1)) * (CONFIG["temp_scale"]-1.0))
        # meta-controller: consider deception selection
        deception, dec_score = deception_select(belief, self.enemy_model_prior, self.info_actions, cost_limit=resources, max_risk=0.5)
        # meta-rule: if high entropy and high vulnerability, favor deceptive behavior
        trigger_deceptive = False
        if be_entropy > 3.5 and opponent_vul > 0.05:
            trigger_deceptive = True
        # adjust local parameters
        local_deception_cost_lambda = CONFIG["deception_cost_lambda"]
        if z == "deceptive" or trigger_deceptive:
            local_deception_cost_lambda = max(0.35, CONFIG["deception_cost_lambda"] * 0.75)
        # resource allocation
        tasks = [{"id":"core","need":30,"value":0.9},{"id":"support","need":20,"value":0.6},{"id":"reserve","need":10,"value":0.3}]
        allocation = resource_optimize(tasks, resources, belief)
        # generate candidate actions from policy with temperature and mode constraints
        candidates=[]
        for _ in range(CONFIG["policy_candidates"]):
            idx, prob, val = self.policy.act(belief, temp=temp)
            a = action_from_index(idx) if isinstance(idx,int) else idx
            # mode-based force cap
            max_force = CONFIG["max_force_by_mode"].get(z, CONFIG["max_force_by_mode"]["balanced"])
            if a.get("type")=="invest" and a.get("force",0) > max_force:
                a["force"] = max_force
            # if deceptive mode triggered, slightly boost innovate value
            if (z=="deceptive" or trigger_deceptive) and a.get("type")=="innovate":
                val += 0.05
            candidates.append((a,prob,val,idx))
        # evaluate candidates with short-term / long-term fusion; adapt alpha by entropy
        alpha = CONFIG["short_term_weight"]
        if be_entropy > 3.0:
            alpha = max(0.15, alpha * 0.7)
        best=None; best_score=-1e9; best_idx=0
        for a,prob,val,idx in candidates:
            if time.time()-start>time_budget: break
            short = mcts_rollout(belief, a, self.simulator_cls, depth=CONFIG["mcts_depth"], rollouts=CONFIG["mcts_rollouts"])
            seq = [a] + [{"type":"invest","force":min(25, a.get("force",20))} for _ in range(6)]
            long_score = system_projection_score({"strength": float(np.dot(STATE_SPACE, belief))}, seq, steps=6)
            # risk penalty: if deception exists and is high risk, penalize by local lambda
            risk_penalty = 0.0
            if deception:
                risk_penalty = local_deception_cost_lambda * deception.cost
            combined = alpha * short + (1.0-alpha) * long_score + 0.2 * val - 0.01 * risk_penalty
            if combined > best_score:
                best_score = combined; best = a; best_idx = idx
        if best is None:
            best = random.choice([c[0] for c in candidates])
            best_idx = 0
        # bookkeeping
        self.metrics["action_counter"][best.get("type")] += 1
        if deception:
            self.metrics["deception_counter"][deception.id] += 1
        # audit high-risk deception
        if deception and deception.risk >= CONFIG["audit_threshold_risk"]:
            self.audit_log.append({"time": timestamp(), "reason": "high_risk_info", "info": deception.as_dict(), "obs": observation, "mode": z})
        decision = {"belief": belief, "deception": deception.as_dict() if deception else None,
                    "allocation": allocation, "action": best, "action_index": best_idx, "dec_score": best_score, "mode": z}
        logging.info(f"DECIDE: mode={z} entropy={be_entropy:.3f} vul={opponent_vul:.4f} obs={observation['strength_est']} action={decision['action']} deception={decision['deception']} alloc={allocation}")
        return decision

    def execute_and_record(self, decision, env, resources):
        if decision["deception"]:
            chosen = next((ia for ia in self.info_actions if ia.id==decision["deception"]["id"]), None)
            if chosen:
                self.enemy_model_prior = simulate_enemy_update(self.enemy_model_prior, chosen)
        obs, reward, done, _ = env.step(decision["action"])
        self.metrics["total_reward"] += float(reward)
        if done and env.true_state.get("strength", None) == 0:
            self.metrics["wins"] += 1
        self.experience_buffer.append(Experience(decision["belief"], decision["action_index"], float(reward), obs, done))
        return obs, float(reward), bool(done)

    def sample_trajectories(self, num_episodes=6, resources=50):
        trajectories = []
        for ep in range(num_episodes):
            env = self.simulator_cls()
            prior = normalize(np.ones(len(STATE_SPACE)))
            done=False; step=0
            states=[]; actions=[]; rewards=[]
            # sample latent mode per trajectory for diversity
            self.current_mode = random.choice(CONFIG["latent_modes"])
            while not done and step<CONFIG["simulate_steps"]:
                obs = env.generate_observation()
                decision = self.decide(prior, obs, resources)
                action_idx = decision["action_index"]
                a = decision["action"]
                obs2, r, done, _ = env.step(a)
                states.append(decision["belief"])
                actions.append(action_idx)
                rewards.append(r)
                prior = decision["belief"]
                step += 1
            if len(states)==0:
                continue
            T = len(rewards)
            values = np.zeros(T+1)
            returns = np.zeros(T)
            adv = np.zeros(T)
            lastgaelam = 0
            for t in reversed(range(T)):
                delta = rewards[t] + CONFIG["gamma"] * values[t+1] - values[t]
                lastgaelam = delta + CONFIG["gamma"] * CONFIG["gae_lambda"] * lastgaelam
                adv[t] = lastgaelam
                returns[t] = adv[t] + values[t]
            trajectories.append({"states": np.vstack(states), "actions": np.array(actions, dtype=int),
                                 "returns": returns, "advantages": adv})
        return trajectories

    def train_policy(self, epochs=3, resources=50, save_prefix="policy"):
        if not HAS_TORCH:
            logging.info("Torch not available: skipping PPO training.")
            return
        for ep in range(epochs):
            trajs = self.sample_trajectories(num_episodes=6, resources=resources)
            if len(trajs)==0:
                continue
            all_adv = np.hstack([t["advantages"] for t in trajs if t["advantages"].size>0])
            if all_adv.size>0:
                mean_adv = np.mean(all_adv); std_adv = np.std(all_adv) + 1e-8
                for t in trajs:
                    if t["advantages"].size>0:
                        t["advantages"] = (t["advantages"] - mean_adv) / std_adv
            self.policy.update(trajs)
            if ep % 2 == 0:
                rules = distill_policy_to_rules(self.policy, sample_states=400)
                fname = os.path.join(CONFIG["log_dir"], f"distilled_rules_{save_prefix}_{timestamp()}.json")
                with open(fname, "w") as f:
                    json.dump(rules, f, indent=2)
                logging.info(f"Distilled rules saved, count={len(rules)}")
            try:
                model_path = os.path.join(CONFIG["model_dir"], f"{save_prefix}_{timestamp()}.pt")
                torch.save(self.policy.net.state_dict(), model_path)
                logging.info(f"Saved model to {model_path}")
            except Exception:
                pass

    def evaluate_across_scenarios(self, prior, resources):
        scenarios = [self.simulator_cls(true_state=max(1,int(np.dot(STATE_SPACE, prior)+random.randint(-20,20))), context=random.choice(["normal","disrupted"])) for _ in range(CONFIG["scenario_count"])]
        scores=[]
        for sc in scenarios:
            prior_local = prior.copy()
            env = sc
            total=0.0
            for step in range(8):
                obs = env.generate_observation()
                dec = self.decide(prior_local, obs, resources)
                obs2, r, done, _ = env.step(dec["action"])
                total += r
                prior_local = dec["belief"]
                if done: break
            scores.append(total)
        return float(np.mean(scores)), float(np.std(scores))

# -------------------------
# Run harness
# -------------------------
def run_training_loop(episodes=200, resources=60, train=False):
    engine = StrategicPinnacle()
    summary = {"episode_metrics": []}
    for ep in range(episodes):
        env = DomainEnv()
        prior = normalize(np.ones(len(STATE_SPACE)))
        done=False; step=0; ep_reward=0.0; action_hist=Counter()
        # sample latent mode for episode (keeps opponents guessing)
        engine.current_mode = random.choice(CONFIG["latent_modes"])
        logging.info(f"Episode {ep+1} start: true_strength={env.true_state['strength']}, context={env.true_state['context']} mode={engine.current_mode}")
        while not done and step<CONFIG["simulate_steps"]:
            obs = env.generate_observation()
            decision = engine.ooda.loop_once(prior, obs, resources)
            logging.info(f"[Step {step}] Obs={obs['strength_est']} Mode={decision.get('mode')} Action={decision['action']} Deception={decision['deception']} Allocation={decision['allocation']}")
            obs2, reward, done, _ = env.step(decision['action'])
            prior = decision['belief']
            engine.metrics["total_reward"] += reward
            ep_reward += reward
            action_hist[decision['action'].get("type")] += 1
            step += 1
        mean_score, std_score = engine.evaluate_across_scenarios(prior, resources)
        logging.info(f"Episode {ep+1} end: true_strength={env.true_state['strength']}, steps={step} ep_reward={ep_reward:.3f} scenario_mean={mean_score:.3f}")
        summary["episode_metrics"].append({"episode": ep+1, "ep_reward": ep_reward, "steps": step, "scenario_mean": mean_score, "scenario_std": std_score, "action_hist": dict(action_hist), "mode": engine.current_mode})
        if train and HAS_TORCH:
            engine.train_policy(epochs=1, resources=resources, save_prefix=f"ep{ep+1}")
        if engine.audit_log and ep % 10 == 0:
            with open(os.path.join(CONFIG["log_dir"], f"audit_{timestamp()}.json"), "w") as f:
                json.dump(engine.audit_log, f, indent=2)
    logging.info("=== Summary Metrics ===")
    logging.info(engine.metrics)
    if HAS_TORCH:
        try:
            rules = distill_policy_to_rules(engine.policy, sample_states=800)
            with open(os.path.join(CONFIG["log_dir"], f"distilled_rules_final_{timestamp()}.json"), "w") as f:
                json.dump(rules, f, indent=2)
            logging.info("Final distilled rules saved.")
        except Exception:
            pass
    return engine, summary

# -------------------------
# Command-line interface
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--resources", type=int, default=60)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean:
        if os.path.exists(CONFIG["log_dir"]):
            shutil.rmtree(CONFIG["log_dir"])
        if os.path.exists(CONFIG["model_dir"]):
            shutil.rmtree(CONFIG["model_dir"])
        os.makedirs(CONFIG["log_dir"], exist_ok=True)
        os.makedirs(CONFIG["model_dir"], exist_ok=True)
        logging.info("Cleaned logs and models directories.")
    run_training_loop(episodes=args.episodes, resources=args.resources, train=args.train)

if __name__ == "__main__":
    main()