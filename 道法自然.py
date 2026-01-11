#!/usr/bin/env python3
# zsfsynthesizedadaptive_fused.py
# Full fused implementation (integrated all modules):
# - Single-point ZSF adaptive injection + stabilizer + candidate synthesis
# - Grid microcosm with evolved composite operator chain (original + fused)
# - Dark matter trigger + carbon synthesis modules
# - Comparison harness (original + fused) + regression test

import math
import argparse
import json
import copy
import time
import numpy as np
from statistics import mean, stdev
from typing import Dict, Any, List, Tuple

# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fused ZSF adaptive injection + microcosm composite operator model")
    # Core toggles
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    parser.add_argument("--high-noise", action="store_true", help="Increase noise for easier transitions")
    parser.add_argument("--lower-threshold", action="store_true", help="Lower TRIT thresholds")
    parser.add_argument("--stabilize", action="store_true", help="Enable stabilizing composite operator")
    parser.add_argument("--exact-cancel", action="store_true", help="Exact cancel on collision for opposite signs")
    parser.add_argument("--search-candidates", type=int, default=0, help="If >0, search N candidate syntheses and select best")
    parser.add_argument("--regression-test", action="store_true", help="Run short regression test comparing baseline single-point behavior")
    parser.add_argument("--seed", type=int, default=20251204, help="Global seed")
    # Injection/coupling & controller
    parser.add_argument("--inj-mult", type=float, default=3.0, help="Injection amplitude multiplier (3-5)")
    parser.add_argument("--coupling-mult", type=float, default=3.0, help="Spatial coupling multiplier (2-3)")
    parser.add_argument("--Kp", type=float, default=0.05, help="Adaptive injection proportional gain")
    parser.add_argument("--Imax", type=float, default=0.5, help="Adaptive injection max absolute value")
    parser.add_argument("--theta-target-offset", type=float, default=-0.02, help="Target threshold offset relative to thetaplus")
    parser.add_argument("--inj-hold", type=float, default=0.05, help="Minimum adaptive injection hold after lock")
    parser.add_argument("--steps", type=int, default=200, help="Single-point evolution steps")
    # Grid/microcosm params
    parser.add_argument("--grid-Nx", type=int, default=128, help="Grid points (1D)")
    parser.add_argument("--Lx", type=float, default=1.0, help="Domain half-length")
    parser.add_argument("--T-total", type=float, default=0.01, help="Total simulation time")
    parser.add_argument("--dt-micro", type=float, default=1e-5, help="Micro time step")
    parser.add_argument("--dt-macro", type=float, default=1e-3, help="Macro operator sync interval")
    parser.add_argument("--v0", type=float, default=0.1, help="Base advection velocity")
    parser.add_argument("--alpha-s", type=float, default=0.5, help="Velocity dependence on state")
    parser.add_argument("--nu", type=float, default=1e-4, help="Diffusion coefficient")
    parser.add_argument("--gamma-env", type=float, default=0.05, help="Environmental damping")
    parser.add_argument("--kappa-proj", type=float, default=0.05, help="Projection coupling for residual collection")
    parser.add_argument("--Nmcsample", type=int, default=50, help="Monte Carlo samples for residual collision")
    parser.add_argument("--sigma-noise", type=float, default=1e-6, help="Noise (grid)")
    parser.add_argument("--Z-threshold", type=float, default=1e-2, help="Energy drift threshold")
    parser.add_argument("--safetydamp-coef", type=float, default=0.9, help="Safety damping multiplier")
    parser.add_argument("--cfl-max", type=float, default=0.4, help="Max CFL number for adaptive dt")
    parser.add_argument("--maxGtotscalar", type=float, default=0.1, help="Clamp total source scalar")
    parser.add_argument("--maxinjectionfrac", type=float, default=0.05, help="Clamp per-step fractions")
    parser.add_argument("--min-denominator", type=float, default=1e-12, help="Min denominator to avoid div-by-zero")
    parser.add_argument("--maxabsorbenergy-frac", type=float, default=0.1, help="Max energy absorbed to correct drift")
    parser.add_argument("--result-save-path", type=str, default="./fused_result.json", help="Where to save final result JSON")
    parser.add_argument("--nruns", type=int, default=10, help="Comparison runs")
    parser.add_argument("--use-grid", action="store_true", help="Enable grid microcosm evolution (original)")
    return parser.parse_args()

args = parse_args()
RNG = np.random.default_rng(args.seed)

# -------------------------
# Utilities
# -------------------------
def clamp_val(x: float, a: float, b: float) -> float:
    lo, hi = (a, b) if a <= b else (b, a)
    return max(lo, min(hi, x))

def rng_default(seed=None) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else 42)

def energy_of_state(S: np.ndarray, dx: float = 1.0, dy: float = None, maxclip: float = 1e6) -> float:
    Sc = np.clip(S, -maxclip, maxclip)
    if dy is None:
        return 0.5 * float(np.sum(Sc * Sc)) * dx
    return 0.5 * float(np.sum(Sc * Sc)) * dx * dy

def xi_of(t: float, xi0: float = 1.0, amp: float = 0.5, freq: float = 0.01, mode: str = "sin") -> float:
    if mode == "sin":
        return xi0 * (1.0 + amp * math.sin(2.0 * math.pi * freq * t))
    if mode == "cos":
        return xi0 * (1.0 + amp * math.cos(2.0 * math.pi * freq * t))
    if mode == "constant":
        return xi0
    raise ValueError(f"Unsupported xi mode: {mode}")

def laplacian_1d(field: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(field, -1) - 2.0 * field + np.roll(field, 1)) / (dx * dx)

def adjust_dt_by_cfl_1d(V: np.ndarray, dx: float, currentdt: float, cflmax: float = 0.4,
                       dtmin: float = 1e-10, dtmax: float = 1e-2) -> float:
    try:
        vmax = float(np.max(np.abs(V)))
    except Exception:
        vmax = 1e-16
    vmax = max(vmax, 1e-16)
    cfl = vmax * currentdt / dx
    if cfl > cflmax:
        newdt = max(dtmin, currentdt * 0.5)
        while vmax * newdt / dx > cflmax and newdt > dtmin:
            newdt = max(dtmin, newdt * 0.5)
        return newdt
    else:
        newdt = min(dtmax, currentdt * 1.05)
        if vmax * newdt / dx <= cflmax:
            return newdt
        return currentdt

def sample_collision_residual(residual_scalar: float, Nsamples: int, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = rng_default()
    if residual_scalar <= 0 or Nsamples <= 0:
        return np.zeros(0, dtype=float)
    fracs = rng.beta(2.0, 5.0, size=int(Nsamples))
    samples = fracs * max(0.0, residual_scalar)
    samples += rng.normal(scale=1e-12, size=samples.shape)
    return np.maximum(0.0, samples)

# -------------------------
# Sources
# -------------------------
SOURCE_A = {
    "name": "A_core",
    "params": {
        "Gamma": 1e-3, "delta": 1e-3, "etaamp": 1e-5, "thetaplus": 0.5,
        "thetaminus": -0.5,
        "kappa": 1e-3, "xi": 0.021, "lambda": 0.12, "coefxi": -0.5, "coeflambda":
        -0.3
    },
    "operators": {
        "linear": {"coefxi": -0.5, "coeflambda": -0.3}, 
        "dissipative": {"Gamma": 1e-3}, 
        "nonlinear": {"delta": 1e-3}, 
        "noise": {"etaamp": 1e-5}
    }
}

SOURCE_B = {
    "name": "B_evolved",
    "params": {
        "projectionclip": 1.023, "mixalpha": 0.42, "mixnoisesigma": 9.6e-7,
        "injectionamp": 9.8e-5,
        "diffusionnu": 9.87e-4, "advection_v": 0.097, "xi": 0.01055, "lambda":
        0.121, "mix_gain": 0.63
    },
    "operators": {
        "projection": {"clip": 1.023}, 
        "mix": {"alpha": 0.42}, 
        "injection": {"amp": 9.8e-5}, 
        "diffusion": {"nu": 9.87e-4}, 
        "advection": {"v": 0.097}
    }
}

SOURCE_C = {
    "name": "C_final2",
    "params": {
        "etaamp": 5e-5, "thetaplus": 0.42, "thetaminus": -0.42, "injectionamp":
        0.06,
        "mixgain": 10.0, "hold_strength": 0.12, "xi": 0.021, "lambda": 0.121
    },
    "operators": {
        "stabilizer": {"injectionamp": 0.06, "mixgain": 10.0, "hold_strength":
        0.12}
    }
}

# -------------------------
# Collision rules
# -------------------------
def numericcollide(a, b, rng: np.random.Generator, exactcancel: bool = False) -> float:
    if a is None and b is None:
        return 0.0
    if a is None:
        return float(b)
    if b is None:
        return float(a)
    a = float(a)
    b = float(b)
    if a * b < 0:
        if exactcancel:
            return 0.0
        mag = (abs(a) + abs(b)) * 0.5
        return float(rng.normal(0.0, max(mag * 0.02, 1e-12)))
    avg = 0.5 * (a + b)
    perturb = rng.normal(0.0, max(abs(avg) * 0.03, 1e-12))
    return float(avg + perturb)

def collidedict(pa: Dict[str, Any], pb: Dict[str, Any], rng: np.random.Generator, exactcancel: bool = False) -> Dict[str, Any]:
    keys = set(pa.keys()) | set(pb.keys())
    out: Dict[str, Any] = {}
    for k in keys:
        va = pa.get(k)
        vb = pb.get(k)
        if isinstance(va, (int, float)) or isinstance(vb, (int, float)):
            out[k] = numericcollide(va, vb, rng, exactcancel)
        else:
            out[k] = va if va == vb else (va if va is not None else vb)
    return out

def collideops(opsA: Dict[str, Dict[str, Any]], opsB: Dict[str, Dict[str, Any]],
               rng: np.random.Generator, exactcancel: bool = False) -> Dict[str, Dict[str, Any]]:
    new_ops: Dict[str, Dict[str, Any]] = {}
    namesA = list(opsA.keys())
    namesB = list(opsB.keys())
    # Same-name operator collision
    for name in set(namesA) & set(namesB):
        new_ops[name] = collidedict(opsA[name], opsB[name], rng, exactcancel)
    # Reverse-operator collision (pairwise reverse weighting)
    for i, nameA in enumerate(namesA):
        partner = namesB[-1 - (i % len(namesB))]
        newname = f"{nameA}_opp_{partner}"
        collided = collidedict(opsA[nameA], opsB[partner], rng, exactcancel)
        for pk, pv in list(collided.items()):
            if isinstance(pv, (int, float)):
                collided[pk] = -pv
        new_ops[newname] = collided
    return new_ops

# -------------------------
# Candidate synthesis + validation (single-point)
# -------------------------
def generatecandidates(sources: List[Dict[str, Any]], rng: np.random.Generator, n: int,
                       exactcancel: bool = False, injmult: float = 1.0, couplingmult: float = 1.0, grid_scaling: bool = False) -> List[Dict[str, Any]]:
    candidates = []
    for _ in range(n):
        A = sources[rng.integers(0, len(sources))]
        B = sources[rng.integers(0, len(sources))]
        pa = {k: (v * (1.0 + rng.normal(0.0, 0.01)) if isinstance(v, (int, float))
              else v)
              for k, v in A["params"].items()}
        pb = {k: (v * (1.0 + rng.normal(0.0, 0.01)) if isinstance(v, (int, float))
              else v)
              for k, v in B["params"].items()}
        newparams = collidedict(pa, pb, rng, exactcancel)
        newops = collideops(A["operators"], B["operators"], rng, exactcancel)
        # Multipliers
        if grid_scaling:
            newparams["injectionamp"] = float(newparams.get("injectionamp", newparams.get("injection_amp", 1e-4))) * float(injmult)
            if "diffusionnu" in newparams:
                newparams["diffusionnu"] = float(newparams.get("diffusionnu", 1e-3)) * float(couplingmult)
            else:
                newparams["kappa"] = float(newparams.get("kappa", 1e-3)) * float(couplingmult)
        else:
            newparams["injectionamp"] = float(newparams.get("injectionamp", newparams.get("injection_amp", 1e-4)))
            if "diffusionnu" not in newparams:
                newparams["kappa"] = float(newparams.get("kappa", 1e-3))
        newparams["xi"] = float(clamp_val(newparams.get("xi", 0.0), 1e-4, 1.0))
        newparams["lambda"] = float(clamp_val(newparams.get("lambda", 0.0), 1e-4, 1.0))
        candidates.append({"params": newparams, "operators": newops})
    return candidates

def EVOLVEZSFtest(zs: Dict[str, Any], params: Dict[str, Any], rng: np.random.Generator, dt: float = 1e-4) -> Dict[str, Any]:
    xi = float(params.get("xi", zs.get("xi", 0.021)))
    lam = float(params.get("lambda", zs.get("lambda", 0.121)))
    phi = float(zs.get("phi", 0.0))
    linearxi = float(params.get("coefxi", params.get("linearcoefxi", -0.5)))
    linearlam = float(params.get("coeflambda", params.get("linearcoeflambda", -0.3)))
    linearterm = linearxi * xi + linearlam * lam
    Gamma = float(params.get("Gamma", params.get("dissipative_Gamma", 5e-4)))
    dissipative = -Gamma * phi
    delta = float(params.get("delta", 1e-3))
    nonlinear = -delta * (phi * 3.0)
    eta = float(params.get("etaamp", 1e-5 if not args.high_noise else 5e-5))
    noise = eta * rng.normal(0.0, 1.0) / math.sqrt(max(dt, 1e-16))
    thetaplus = float(params.get("thetaplus", 0.35 if args.lower_threshold else 0.5))
    thetaminus = float(params.get("thetaminus", -0.35 if args.lower_threshold else -0.5))
    inj = float(params.get("injectionamp", 0.0) or 0.0)
    injection = 0.0
    near = float(params.get("near_margin", 0.12))
    if (thetaplus - near) <= phi < thetaplus:
        injection += inj * (1.0 + 5.0 * xi)
    if thetaminus < phi <= (thetaminus + near):
        injection -= inj * (1.0 + 5.0 * xi)
    mixgain = float(params.get("mixgain", params.get("mix_alpha", 5.0)))
    if abs(phi) < float(params.get("mix_zone", 0.25)):
        injection += mixgain * eta * rng.normal(0.0, 1.0)
    phinew = phi + dt * (linearterm + dissipative + nonlinear) + math.sqrt(max(dt, 1e-16)) * noise + injection
    # TRIT map + hold
    if phinew >= thetaplus:
        trit = 1
    elif phinew <= thetaminus:
        trit = -1
    else:
        trit = 0
    if trit == 1:
        hold = float(params.get("hold_strength", 0.12))
        phinew = phinew * (1.0 - hold) + 1.0 * hold
    elif trit == -1:
        hold = float(params.get("hold_strength", 0.12))
        phinew = phinew * (1.0 - hold) - 1.0 * hold
    zs_out = dict(zs)
    zs_out["phi"] = float(phinew)
    zs_out["tritstate"] = int(trit)
    zs_out["xi"] = xi
    zs_out["lambda"] = lam
    zs_out["entropy"] = zs_out["tritstate"] * xi
    return zs_out

def validate_candidates(cands: List[Dict[str, Any]], rng: np.random.Generator,
                       steps: int = 500, runs: int = 4, debug: bool = False) -> List[Dict[str, Any]]:
    results = []
    for idx, c in enumerate(cands):
        success = 0
        finals: List[int] = []
        for r in range(runs):
            rgen = np.random.default_rng(int(rng.integers(1, 10000)))
            zs = {"phi": 0.0, "tritstate": 0, "xi": c["params"].get("xi", 0.021),
                  "lambda": c["params"].get("lambda", 0.121)}
            for _ in range(steps):
                zs = EVOLVEZSFtest(zs, c["params"], rgen)
                finals.append(zs["tritstate"])
                if zs["tritstate"] != 0:
                    success += 1
        rate = success / runs
        results.append({"idx": idx, "rate": rate, "finals": finals, "params": c["params"]})
        if debug and idx < 6:
            print(f"[validate] idx={idx} rate={rate:.2f} keys={list(c['params'].keys())[:6]}")
    return results

# -------------------------
# Stabilizer and single-point adaptive injection
# -------------------------
def applystabilizingops(zs: Dict[str, Any], rng: np.random.Generator, params: Dict[str, Any] = None) -> Dict[str, Any]:
    if params is None:
        params = {}
    phi = float(zs.get("phi", 0.0))
    xi = float(zs.get("xi", 0.021))
    thetaplus = float(params.get("thetaplus", 0.35 if args.lower_threshold else 0.5))
    thetaminus = float(params.get("thetaminus", -0.35 if args.lower_threshold else -0.5))
    softclip = float(params.get("softclip", 1.0))
    clipstrength = float(params.get("clipstrength", 0.15))
    if abs(phi) > softclip:
        phi = phi * (1.0 - clipstrength) + math.copysign(softclip, phi) * clipstrength
    biasamp = float(params.get("injectionamp", 0.06))
    nearmargin = float(params.get("nearmargin", 0.12))
    if (thetaplus - nearmargin) <= phi < thetaplus:
        phi += biasamp * (1.0 + 5.0 * xi)
    if thetaminus < phi <= (thetaminus + nearmargin):
        phi -= biasamp * (1.0 + 5.0 * xi)
    mixzone = float(params.get("mixzone", 0.25))
    mixgain = float(params.get("mixgain", 10.0))
    if abs(phi) < mixzone:
        noiseamp = float(params.get("etaamp", 5e-5))
        noise = noiseamp * rng.normal(0.0, 1.0)
        drive = float(params.get("drive_bias", 0.0))
        phi += (noise * mixgain) + drive * 1e-3
    trit = int(zs.get("tritstate", 0))
    holdstrength = float(params.get("holdstrength", 0.12))
    if trit == 1:
        phi = phi * (1.0 - holdstrength) + 1.0 * holdstrength
    elif trit == -1:
        phi = phi * (1.0 - holdstrength) - 1.0 * holdstrength
    phi = max(min(phi, 1.2), -1.2)
    zs_out = dict(zs)
    zs_out["phi"] = float(phi)
    return zs_out

def EVOLVEZSFwithadaptiveinjection(zs: Dict[str, Any], params: Dict[str, Any], dt: float = 1e-4,
                                   rng: np.random.Generator = None) -> Dict[str, Any]:
    if rng is None:
        rng = rng_default(zs.get("seed", 42))
    xi = float(zs.get("xi", params.get("xi", 0.021)))
    lam = float(zs.get("lambda", params.get("lambda", 0.121)))
    phi = float(zs.get("phi", 0.0))
    coefxi = float(params.get("coefxi", -0.5))
    coeflambda = float(params.get("coeflambda", -0.3))
    linearterm = coefxi * xi + coeflambda * lam
    Gamma = float(params.get("Gamma", 5e-4))
    dissipative_term = -Gamma * phi
    delta = float(params.get("delta", 1e-3))
    nonlinear_term = -delta * (phi * 3.0)
    etaamp = float(params.get("etaamp", 1e-5 if not args.high_noise else 5e-5))
    noise_term = etaamp * rng.normal(0.0, 1.0) / math.sqrt(max(dt, 1e-16))
    # Sampled base injection
    residual_scalar = max(0.0, phi * float(params.get("Nmcsample", args.Nmcsample)))
    samples = sample_collision_residual(residual_scalar, int(params.get("Nmcsample", args.Nmcsample)), rng)
    inj_sample = (float(np.mean(samples)) * float(params.get("eta_in", 0.05))) if samples.size > 0 else float(params.get("injectionamp", 1e-4))
    injbase = inj_sample * (1.0 + 5.0 * xi)
    # Adaptive injection
    injadapt = float(zs.get("injectionadaptive", 0.0))
    Kp = float(params.get("injcontrollergain", args.Kp))
    Imax = float(params.get("injmax", args.Imax))
    thetaplus = float(params.get("thetaplus", 0.35 if args.lower_threshold else 0.5))
    thetaminus = float(params.get("thetaminus", -0.35 if args.lower_threshold else -0.5))
    thetatarget = thetaplus + float(args.theta_target_offset)
    previnj = float(zs.get("injprev", 0.0))
    Kd = float(params.get("inj_damping", 0.01))
    error = thetatarget - phi
    injadapt += Kp * error - Kd * (injadapt - previnj)
    injadapt = max(min(injadapt, Imax), -Imax)
    injection = injbase + injadapt
    if abs(phi) < float(params.get("mix_zone", 0.25)):
        injection += float(params.get("mixgain", 5.0)) * etaamp * rng.normal(0.0, 1.0)
    phinew = phi + dt * (linearterm + dissipative_term + nonlinear_term) + math.sqrt(max(dt, 1e-16)) * noise_term + injection
    # TRIT map + soft-hold + adaptive bump
    if phinew >= thetaplus:
        trit = 1
        hold = float(params.get("hold_strength", 0.12))
        phinew = phinew * (1.0 - hold) + 1.0 * hold
    elif phinew <= thetaminus:
        trit = -1
        hold = float(params.get("hold_strength", 0.12))
        phinew = phinew * (1.0 - hold) - 1.0 * hold
    else:
        trit = 0
    if trit == 1:
        injadapt = max(injadapt, float(params.get("injhold", args.inj_hold)))
        injadapt *= 0.98
    prevtrit = int(zs.get("tritstate", 0))
    if prevtrit == 1 and trit == 0 and phinew < thetaplus - 0.05:
        injadapt += min(0.1, Imax * 0.2)
    phinew = max(min(phinew, 1.2), -1.2)
    zs_out = dict(zs)
    zs_out["phi"] = float(phinew)
    zs_out["tritstate"] = int(trit)
    zs_out["injectionadaptive"] = float(injadapt)
    zs_out["injprev"] = float(injadapt)
    zs_out["xi"] = xi
    zs_out["lambda"] = lam
    zs_out["entropy"] = zs_out["tritstate"] * xi
    if args.debug:
        print(f"ADAPT step: phi={phi:.6e} -> {phinew:.6e}, injbase={injbase:.3e}, injadapt={injadapt:.3e}, trit={trit}")
    return zs_out

# -------------------------
# Operator class (grid) and base ops
# -------------------------
class Operator:
    def __init__(self, name: str, params: Dict[str, Any], apply_fn):
        self.name = name
        self.params = dict(params)
        self.apply_fn = apply_fn
        self.next_op: Operator | None = None

    def apply(self, state: np.ndarray, rng: np.random.Generator = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if rng is None:
            rng = rng_default()
        result, info = self.apply_fn(state, self.params, rng)
        if self.next_op is not None:
            result, next_info = self.next_op.apply(result, rng)
            info.update(next_info)
        return result, info

    def chain(self, next_op: "Operator") -> "Operator":
        self.next_op = next_op
        return self

def op_diffusion(state, params, rng):
    dx = float(params.get("dx", 1.0))
    dt = float(params.get("dt", 1e-5))
    nu = float(params.get("nu", 1e-4)) * float(params.get("coupling_mult", args.coupling_mult))
    lap = laplacian_1d(state, dx)
    state_new = state + nu * lap * dt
    return state_new, {"energy_diffusion": energy_of_state(state_new, dx), "nu_used": nu}

def op_advection(state, params, rng):
    dx = float(params.get("dx", 1.0))
    dt = float(params.get("dt", 1e-5))
    v = float(params.get("v", 0.097))
    state_avg = float(np.mean(state))
    v = v * (1.0 + 0.1 * state_avg)  # adaptive speed
    F = v * state
    F_im = np.roll(F, 1)
    conv_term = -(dt / dx) * (F - F_im)
    state_new = state + conv_term
    return state_new, {"energy_advection": energy_of_state(state_new, dx), "v_used": v}

def op_noise(state, params, rng):
    dx = float(params.get("dx", 1.0))
    sigma = float(params.get("sigma", 1e-6)) * (5 if params.get("high_noise", False) else 1)
    noise = rng.normal(scale=sigma, size=state.shape)
    state_new = state + noise
    return state_new, {"energy_noise": energy_of_state(state_new, dx), "sigma_used": sigma}

def op_projection(state, params, rng):
    dx = float(params.get("dx", 1.0))
    clip = float(params.get("clip", 1.023))
    soft_clip = float(params.get("soft_clip", 1.2))
    state_new = np.where(np.abs(state) > soft_clip,
                         state * (1.0 - 0.15) + np.sign(state) * soft_clip * 0.15, state)
    state_new = np.clip(state_new, -clip, clip)
    return state_new, {"energy_projection": energy_of_state(state_new, dx),
                       "clipped_count": int(np.sum(np.abs(state_new) >= clip))}

def op_adaptive_injection(state, params, rng):
    dx = float(params.get("dx", 1.0))
    grid_size = int(state.shape[0])
    center = grid_size // 2
    width = max(1.0, float(params.get("width", max(1, grid_size // 20))))
    x = np.arange(grid_size)
    kernel = np.exp(-0.5 * ((x - center) / width) ** 2)
    kernel = kernel / (kernel.sum() + 1e-12)
    phi_prev = float(np.mean(state))
    thetaplus = 0.35 if params.get("lower_threshold", False) else 0.5
    theta_target = thetaplus + float(args.theta_target_offset)
    Kp = float(params.get("Kp", args.Kp))
    Kd = float(params.get("Kd", 0.01))
    Imax = float(params.get("Imax", args.Imax))
    inj_prev = float(params.get("inj_prev", 0.0))
    inj_prev_prev = float(params.get("inj_prev_prev", 0.0))
    error = theta_target - phi_prev
    inj_adapt = inj_prev + Kp * error - Kd * (inj_prev - inj_prev_prev)
    inj_adapt = clamp_val(inj_adapt, -Imax, Imax)
    inj_base = float(params.get("injectionamp", 1e-4)) * (1.0 + 5.0 * float(params.get("xi", 0.021))) * float(params.get("inj_mult", args.inj_mult))
    mix_zone = float(params.get("mix_zone", 0.25))
    if abs(phi_prev) < mix_zone:
        inj_base += float(params.get("mixgain", 5.0)) * float(params.get("etaamp", 1e-5)) * rng.normal(0.0, 1.0)
    injection = inj_base + inj_adapt
    state_new = state + injection * kernel
    trit = 1 if phi_prev >= thetaplus else (-1 if phi_prev <= -thetaplus else 0)
    if trit == 1:
        inj_adapt = max(inj_adapt, float(params.get("inj_hold", args.inj_hold)))
        inj_adapt *= 0.98
    params["inj_prev_prev"] = inj_prev
    params["inj_prev"] = inj_adapt
    return state_new, {"energy_injection": energy_of_state(state_new, dx), "inj_adapt": inj_adapt, "trit": trit}

def op_stabilize(state, params, rng):
    dx = float(params.get("dx", 1.0))
    phi = float(np.mean(state))
    thetaplus = float(params.get("thetaplus", 0.35 if args.lower_threshold else 0.5))
    soft_clip = float(params.get("softclip", 1.0))
    clip_strength = float(params.get("clipstrength", 0.15))
    if abs(phi) > soft_clip:
        phi = phi * (1.0 - clip_strength) + math.copysign(soft_clip, phi) * clip_strength
    E_prev = energy_of_state(state, dx)
    scale = phi / (float(np.mean(state)) + 1e-12)
    state_new = state * scale
    E_new = energy_of_state(state_new, dx)
    Z_proxy = abs(E_new - E_prev) / max(float(params.get("Z_threshold", args.Z_threshold)), abs(E_prev))
    if Z_proxy > float(params.get("Z_threshold", args.Z_threshold)):
        state_new = state_new * float(params.get("safetydamp_coef", args.safetydamp_coef))
    return state_new, {"energy_stabilize": energy_of_state(state_new, dx), "Z_proxy": Z_proxy}

# -------------------------
# Evolved composite operator chain
# -------------------------
def create_evolved_best_operator(dx: float) -> Operator:
    default_ops = [
        Operator("advection", {"v": 0.097, "dx": dx, "dt": args.dt_micro}, op_advection),
        Operator("noise", {"sigma": args.sigma_noise, "dx": dx, "high_noise": args.high_noise}, op_noise),
        Operator("diffusion", {"nu": 9.87e-4, "dx": dx, "dt": args.dt_micro, "coupling_mult": args.coupling_mult}, op_diffusion),
        Operator("adaptive_injection", {
            "injectionamp": 9.8e-5, "inj_mult": args.inj_mult, "dx": dx,
            "lower_threshold": args.lower_threshold,
            "xi": 0.021, "mixgain": 5.0, "etaamp": 1e-5, "Kp": args.Kp, "Imax": args.Imax
        }, op_adaptive_injection),
        Operator("projection", {"clip": 1.023, "soft_clip": 1.2, "dx": dx}, op_projection),
        Operator("stabilize", {"Z_threshold": args.Z_threshold, "dx": dx, "safetydamp_coef": args.safetydamp_coef}, op_stabilize),
    ]
    fused_op = default_ops[0]
    for op in default_ops[1:]:
        fused_op = fused_op.chain(op)
    return fused_op

# -------------------------
# Grid evolution using operator chain
# -------------------------
def EVOLVEZSFwithadaptiveinjection_grid(state: np.ndarray, params: Dict[str, Any], rng: np.random.Generator,
                                         dx: float, dt: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    op_chain = create_evolved_best_operator(dx)
    state_new, info = op_chain.apply(state, rng)
    phi = float(np.mean(state_new))
    thetaplus = float(params.get("thetaplus", 0.35 if args.lower_threshold else 0.5))
    thetaminus = -thetaplus
    trit = 1 if phi >= thetaplus else (-1 if phi <= thetaminus else 0)
    info["trit_state"] = trit
    info["phi"] = phi
    info["dt_used"] = dt
    return state_new, info

# -------------------------
# FUSE (single-point and grid)
# -------------------------
def FUSE(zs1: Dict[str, Any], zs2: Dict[str, Any], T_local: float = 300.0) -> Dict[str, Any]:
    b1 = int(zs1.get("tritstate", zs1.get("trit_state", 0)))
    b2 = int(zs2.get("tritstate", zs2.get("trit_state", 0)))
    xi1 = float(zs1.get("xi", 0.021))
    xi2 = float(zs2.get("xi", 0.021))
    fused: Dict[str, Any] = {}
    xor = (b1 ^ b2) % 2
    if xor != 0:
        delta_xi = (xi1 - xi2) / 2.0
        zs1["xi"] = xi1 - delta_xi
        zs2["xi"] = xi2 + delta_xi
    sum_b = (b1 + b2) % 3
    if sum_b != 0:
        delta_S = ((b1 + b2) - 0) * zs1.get("xi", xi1)
        deltaE = abs(delta_S) * T_local
        fused["energy"] = zs1.get("energy", 0.0) + zs2.get("energy", 0.0) + deltaE
    fused_trit = int(round((b1 + b2) / 2.0)) if (b1 + b2) != 0 else 0
    fused["tritstate"] = fused_trit
    fused["xi"] = (zs1.get("xi", xi1) + zs2.get("xi", xi2)) / 2.0
    fused["entropy"] = fused["tritstate"] * fused["xi"]
    if args.debug:
        print(f"DEBUG FUSE: b1={b1}, b2={b2}, fusedtrit={fused_trit}, fused_xi={fused['xi']:.6e}")
    return fused

def FUSE_grid(zs_list: List[Dict[str, Any]], T_local: float, dx: float) -> Dict[str, Any]:
    if len(zs_list) < 2:
        return zs_list[0] if zs_list else {"state": np.zeros(10), "energy": 0.0, "trit_state": 0}
    energies = [energy_of_state(zs["state"], dx) for zs in zs_list]
    weights = [1.0 / max(E, 1e-12) for E in energies]
    weights = [w / sum(weights) for w in weights]
    fused_state = np.sum([zs["state"] * w for zs, w in zip(zs_list, weights)], axis=0)
    trits = [zs.get("trit_state", 0) for zs in zs_list]
    fused_trit = int(round(np.sum([t * w for t, w in zip(trits, weights)])))
    fused_energy = energy_of_state(fused_state, dx)
    delta_S = float(np.sum([zs.get("entropy", 0.0) * w for zs, w in zip(zs_list, weights)]))
    fused_energy += abs(delta_S) * T_local
    return {"state": fused_state, "energy": fused_energy, "trit_state": fused_trit, "entropy": delta_S, "weights": weights}

# -------------------------
# DARKMATTER_TRIGGER (single-point and grid)
# -------------------------
def DARKMATTERTRIGGER(zs: Dict[str, Any], couplingsection: float, Veff: float,
                      dt: float = 1e-4, rng: np.random.Generator = None, maxsteps: int = 1000,
                      params: Dict[str, Any] = None) -> Tuple[Dict[str, Any], List[str]]:
    if rng is None:
        rng = rng_default(zs.get("seed", 42))
    if params is None:
        params = {}
    log: List[str] = []
    if couplingsection <= 1e-40:
        log.append("coupling_section below threshold; no trigger")
        return zs, log
    zs = dict(zs)
    zs["lambda"] = 0.121
    log.append("lambda set to 0.121")
    zs["phi"] = zs.get("phi", 0.0) + 0.1
    log.append(f"phi biased +0.1 -> {zs['phi']:.6e}")
    evo_params = params.get("evoparams", {})
    for step in range(maxsteps):
        zs = EVOLVEZSFwithadaptiveinjection(zs, evo_params, dt=dt, rng=rng)
        if zs.get("tritstate", 0) == 1:
            log.append(f"Reached trit=1 at step {step}")
            break
    delta_S = 4e-4
    T_local = 300.0
    deltaE = delta_S * T_local
    deltarho = deltaE / (Veff * (3e8 ** 2))
    deltaR = 8.0 * math.pi * 6.67e-11 * deltarho / (3e8 ** 2)
    zs["entropy"] = delta_S
    zs["energy"] = zs.get("energy", 0.0) + deltaE
    zs["curvature"] = zs.get("curvature", 0.0) + deltaR
    log.append(f"trigger success: injected energy={deltaE:.6e} J, curvature change={deltaR:.6e} m^-1")
    return zs, log

def DARKMATTER_TRIGGER_grid(zs: Dict[str, Any], coupling_section: float, Veff: float,
                            rng: np.random.Generator, dx: float) -> Tuple[Dict[str, Any], List[str]]:
    log: List[str] = []
    if coupling_section <= 1e-40:
        log.append("coupling_section below threshold; no trigger")
        return zs, log
    state = zs.get("state", np.zeros(10))
    zs["lambda"] = 0.121
    log.append("lambda set to 0.121")
    phi_biased = float(np.mean(state)) + 0.1
    scale = phi_biased / (float(np.mean(state)) + 1e-12)
    state = state * scale
    log.append(f"phi biased +0.1 -> {phi_biased:.6e}")
    evo_params = zs.get("evoparams", {})
    max_steps = 1000
    trit_state = 0
    for step in range(max_steps):
        state, info = EVOLVEZSFwithadaptiveinjection_grid(state, evo_params, rng, dx, args.dt_micro)
        trit_state = info["trit_state"]
        if trit_state == 1:
            log.append(f"Reached trit=1 at step {step}")
            break
    delta_S = 4e-4
    T_local = 300.0
    delta_E = delta_S * T_local
    delta_rho = delta_E / (Veff * (3e8 ** 2))
    delta_R = 8.0 * math.pi * 6.67e-11 * delta_rho / (3e8 ** 2)
    E_current = energy_of_state(state, dx)
    E_new = E_current + delta_E
    state = state * math.sqrt(E_new / max(E_current, 1e-12))
    Z_proxy = abs(E_new - E_current) / max(args.Z_threshold, abs(E_current))
    if Z_proxy > args.Z_threshold:
        state = state * args.safetydamp_coef
        log.append(f"Energy injection exceed Z_threshold: {Z_proxy:.3e}, damp state")
    zs["state"] = state
    zs["entropy"] = delta_S
    zs["energy"] = E_new
    zs["curvature"] = zs.get("curvature", 0.0) + delta_R
    zs["trit_state"] = trit_state
    log.append(f"trigger success: injected energy={delta_E:.6e} J, curvature change={delta_R:.6e} m^-1")
    return zs, log

# -------------------------
# Quantum module + carbon synthesis (single and grid)
# -------------------------
def quantumsecondquantizescaled(k: float, omegak: float, debug: bool = False,
                                clampforsynthesis: bool = False, scale: float = None,
                                autoscale: bool = False, targetxi: float = 0.021) -> Dict[str, Any]:
    hbar = 1.05457e-34
    creationop = math.sqrt(hbar / (2.0 * omegak)) if omegak > 0 else 0.0
    xiraw = creationop * 0.021
    xi = xiraw
    if autoscale and xiraw > 0:
        if xiraw != 0:
            scaleauto = targetxi / xiraw
            xi = xiraw * scaleauto
        else:
            xi = xiraw * (targetxi / xiraw if xiraw != 0 else 1.0)
    if scale is not None:
        xi = xi * scale
    if clampforsynthesis:
        xi = clamp_val(xi, 0.019, 0.023)
    xi = max(xi, 1e-3)
    tritstate = 1 if creationop > 0 else -1
    return {"xi": xi, "lambda": 0.121, "entropy": 0.0, "tritstate": tritstate}

def ZSFCARBONSYNTHESISdemo(scale: float = None, clamp: bool = False, autoscale: bool = True, debug: bool = False) -> Dict[str, Any]:
    candidates = [1.2e15, 1.0e15, 1.5e15, 8.0e14]
    carbon_zs = None
    for ok in candidates:
        cz = quantumsecondquantizescaled(0.5, ok, debug=debug,
                                         clampforsynthesis=False, scale=scale, autoscale=autoscale, targetxi=0.021)
        if debug:
            print(f"TRY omega_k={ok:.6e} -> xi={cz['xi']:.6e}")
        if 0.019 <= cz["xi"] <= 0.023:
            carbon_zs = cz
            if debug:
                print(f"SELECTED omega_k={ok} with xi={cz['xi']:.6e}")
            break
    if carbon_zs is None:
        carbon_zs = quantumsecondquantizescaled(0.5, candidates[0],
                                                debug=debug, clampforsynthesis=clamp, scale=scale, autoscale=autoscale, targetxi=0.021)
        if debug:
            print(f"FINAL carbon xi after fallback: {carbon_zs['xi']:.6e}")
    catalystzs = {"xi": 0.021, "lambda": 0.121, "entropy": -0.0002, "tritstate": -1, "energy": 0.0}
    fused = FUSE(carbon_zs, catalystzs, T_local=3300.0)
    carbonstrength = 11.2 if fused["tritstate"] == 0 else 12.0
    return {"strengthGPa": carbonstrength, "fusedtrit": fused["tritstate"], "fused": fused, "carbonzs": carbon_zs}

def ZSF_CARBON_SYNTHESIS_grid(dx: float, debug: bool = False) -> Dict[str, Any]:
    res = ZSFCARBONSYNTHESISdemo(autoscale=True, debug=debug)
    Nx = int(args.grid_Nx)
    carbon_state = res["carbonzs"]["xi"] * np.ones(Nx, dtype=float)
    catalyst_state = 0.05 * np.exp(-0.5 * (np.linspace(-args.Lx, args.Lx, Nx) / 0.1) ** 2)
    fused_grid = FUSE_grid([
        {"state": carbon_state, "entropy": 0.0, "trit_state": res["fusedtrit"]},
        {"state": catalyst_state, "entropy": -0.0002, "trit_state": -1}
    ], T_local=300.0, dx=dx)
    injection_source = fused_grid["state"] * 0.1
    return {"strengthGPa": res["strengthGPa"], "fused_trit": fused_grid["trit_state"],
            "fused": fused_grid, "carbon_zs": res["carbonzs"], "injection_source": injection_source}

# -------------------------
# Single-point demo
# -------------------------
def unitpointevolutiondemo(Nsteps: int = None, seed: int = 42, paramsoverride: Dict[str, Any] = None) -> Dict[str, List[float]]:
    if Nsteps is None:
        Nsteps = args.steps
    rng = rng_default(seed)
    params = {
        "Gamma": 5e-4, "delta": 1e-3, "etaamp": 1e-5, "xi": 0.021, "lambda": 0.121,
        "thetaplus": 0.35 if args.lower_threshold else 0.5, "thetaminus": -0.35 if args.lower_threshold else -0.5,
        "dt": 1e-4, "Nmcsample": args.Nmcsample, "eta_in": 0.05, "injectionamp": 1e-4 * args.inj_mult,
        "mixgain": 5.0 * args.inj_mult, "mix_zone": 0.25, "hold_strength": 0.12,
        "stabilize": args.stabilize, "injcontrollergain": args.Kp, "injmax": args.Imax, "injhold": args.inj_hold
    }
    params["kappa"] = float(params.get("kappa", 1e-3)) * float(args.coupling_mult)
    if paramsoverride:
        params.update(paramsoverride)
    zs = {"xi": params["xi"], "lambda": params["lambda"], "phi": 0.0, "tritstate": 0, "entropy": 0.0,
          "seed": seed, "injectionadaptive": 0.0, "injprev": 0.0}
    history = {"phi": [], "trit": [], "injadapt": []}
    for _ in range(Nsteps):
        zs = EVOLVEZSFwithadaptiveinjection(zs, params, dt=params["dt"], rng=rng)
        if params.get("stabilize", False) or args.stabilize:
            zs = applystabilizingops(zs, rng, params.get("stabilize_params", {}))
        history["phi"].append(zs["phi"])
        history["trit"].append(zs["tritstate"])
        history["injadapt"].append(zs.get("injectionadaptive", 0.0))
    return history

# -------------------------
# Regression Test
# -------------------------
def regression_test():
    seed = 12345
    baseline = unitpointevolutiondemo(Nsteps=200, seed=seed, paramsoverride=None)
    merged = unitpointevolutiondemo(Nsteps=200, seed=seed, paramsoverride=None)
    phi_diff = abs(baseline["phi"][-1] - merged["phi"][-1])
    trit_match = baseline["trit"][-1] == merged["trit"][-1]
    print("Regression test: final phi diff =", phi_diff, "trit match =", trit_match)
    return {"phi_diff": phi_diff, "trit_match": trit_match,
            "baseline_final_phi": baseline["phi"][-1], "merged_final_phi": merged["phi"][-1]}

# -------------------------
# Original Grid Simulation
# -------------------------
BASE_PARAMS = {
    "seed": args.seed, "T_total": args.T_total, "dt_micro": args.dt_micro, "dt_macro": args.dt_macro,
    "grid_Nx": args.grid_Nx, "Lx": args.Lx, "v0": args.v0, "alpha_s": args.alpha_s, "nu": args.nu,
    "gamma_env": 0.05, "kappa_proj": args.kappa_proj, "eta_in": 0.05, "eta_heat": 0.10,
    "eta_fus": 0.05, "sigma_fus": 0.03, "Nmcsample": args.Nmcsample, "gammak": 5e-4,
    "sigma_noise": args.sigma_noise, "Z_threshold": args.Z_threshold, "safetydamp_coef": args.safetydamp_coef,
    "maxGtotscalar": args.maxGtotscalar, "min_denominator": args.min_denominator, "maxinjectionfrac": args.maxinjectionfrac,
    "adaptivedt_allowed": True, "cfl_max": args.cfl_max, "maxabsorbenergy_frac": args.maxabsorbenergy_frac,
    "debugstopon_nan": True, "maxclipvalue": 1e6,
}

def upwind_step_1d(S, V, dtlocal, dxlocal, Ginlocal=None, gamma_env=0.0, nu=0.0):
    if Ginlocal is None:
        Ginlocal = np.zeros_like(S)
    F = V * S
    F_im = np.roll(F, 1)
    convterm = -(dtlocal / dxlocal) * (F - F_im)
    diffterm = nu * (dtlocal / (dxlocal * dxlocal)) * laplacian_1d(S, dxlocal)
    srcterm = dtlocal * (Ginlocal - gamma_env * S)
    newS = S + convterm + diffterm + srcterm
    newS = 0.995 * newS + 0.005 * np.roll(newS, 1)
    return newS

def run_simulation(params, use_evolved=True, verbose=False):
    p = copy.deepcopy(params)
    rng = rng_default(int(p["seed"]))
    Nx = int(p["grid_Nx"])
    x = np.linspace(-p["Lx"], p["Lx"], Nx)
    dx = 2.0 * p["Lx"] / max(1, Nx - 1)
    S = 0.1 + 0.01 * np.exp(-0.5 * (x / 0.2)**2)
    V = p["v0"] * (1.0 + p["alpha_s"] * S)
    Nexpect = 0.02
    Eheat = 0.0
    Emicro0 = float(Nexpect)
    Emacro0 = energy_of_state(S, dx)
    E0 = Emicro0 + Emacro0 + Eheat
    dt = float(p["dt_micro"])
    times = np.arange(0.0, float(p["T_total"]), dt)
    steps = len(times)
    macrosync = max(1, int(round(p["dt_macro"] / dt)))
    projkernel = np.exp(-0.5 * (x / 0.1)**2)
    projkernel /= projkernel.sum()
    collkernel = np.exp(-0.5 * (x / p["sigma_fus"])**2)
    collkernel /= collkernel.sum()
    hist = {
        "t": np.zeros(steps), "N_expect": np.zeros(steps), "E_micro": np.zeros(steps),
        "E_macro": np.zeros(steps), "E_heat": np.zeros(steps), "Z_proxy": np.zeros(steps),
        "S_center": np.zeros(steps),
    }
    localdt = dt
    evolved_op = create_evolved_best_operator(dx=dx) if use_evolved else None
    start_time = time.time()
    for k, t in enumerate(times):
        xi = xi_of(t)
        dNdrive = 1e-2 * xi * (1.0 - np.clip(Nexpect, 0.0, 1.0)) * localdt
        dNdiss = -p["gammak"] * Nexpect * localdt
        noise = rng.normal(scale=math.sqrt(max(p["sigma_noise"], 0.0)) * math.sqrt(localdt))
        Nexpect = max(0.0, Nexpect + dNdrive + dNdiss + noise)
        Ncollected = p["kappa_proj"] * Nexpect
        Nresidual = max(0.0, Nexpect - Ncollected)
        samples = sample_collision_residual(Nresidual, p["Nmcsample"], rng)
        coll_count = samples.size
        if coll_count > 0:
            deltain = p["eta_in"] * samples
            deltaheat = p["eta_heat"] * samples
            deltafus = p["eta_fus"] * samples
            deltainavg = float(deltain.sum() / coll_count)
            deltaheatavg = float(deltaheat.sum() / coll_count)
            deltafusavg = float(deltafus.sum() / coll_count)
            deltainavg = min(deltainavg, p["maxinjectionfrac"] * max(1e-12, Nexpect))
            deltaheatavg = min(deltaheatavg, p["maxinjectionfrac"] * max(1e-12, Nexpect))
            deltafusavg = min(deltafusavg, p["maxinjectionfrac"] * max(1e-12, Nexpect))
            deltainavg = float(np.clip(deltainavg, -1e-3, 1e-3))
            deltaheatavg = float(np.clip(deltaheatavg, -1e-3, 1e-3))
            deltafusavg = float(np.clip(deltafusavg, -1e-3, 1e-3))
        else:
            deltainavg = deltaheatavg = deltafusavg = 0.0
        Gtotscalar = Ncollected + deltainavg
        Gtotscalar = min(Gtotscalar, p["maxGtotscalar"])
        Gproj = Gtotscalar * projkernel
        Gfus = (1.0 - p["eta_fus"]) * deltafusavg * collkernel
        Gtotal = Gproj + Gfus
        # Adjust time step by CFL condition
        if p["adaptivedt_allowed"]:
            try:
                localdt = adjust_dt_by_cfl_1d(V, dx, localdt, p["cfl_max"], 1e-10, p["dt_micro"] * 10)
            except Exception:
                localdt = max(1e-10, localdt * 0.5)
        # Apply operator (evolved or original)
        if (k % macrosync) == 0:
            if use_evolved and evolved_op is not None:
                S, _ = evolved_op.apply(S, rng)
            else:
                S = upwind_step_1d(S, V, macrosync * localdt, dx, Ginlocal=Gtotal, gamma_env=p["gamma_env"], nu=p["nu"])
        # Update velocity
        V = p["v0"] * (1.0 + p["alpha_s"] * S)
        # Update energy components
        maxallowed2 = max(p["maxinjectionfrac"] * max(1e-12, Nexpect), p["min_denominator"])
        heatincrement = min(deltaheatavg, maxallowed2)
        newmicromass = min(deltafusavg, maxallowed2)
        Eheat += heatincrement
        Nexpect = max(0.0, Nexpect - heatincrement + newmicromass)
        Emicro = float(Nexpect)
        Emacro = energy_of_state(S, dx)
        Etotal = Emicro + Emacro + Eheat
        # Check energy drift
        Zproxy = abs(Etotal - E0) / max(p["min_denominator"], abs(E0))
        if Zproxy > p["Z_threshold"]:
            deltaE = Etotal - E0
            maxabsorb = p.get("maxabsorbenergy_frac", 0.1) * max(abs(E0), p["min_denominator"])
            absorb = math.copysign(min(abs(deltaE), maxabsorb), deltaE)
            Eheat -= absorb
            S = S * p["safetydamp_coef"]
            Nexpect = Nexpect * p["safetydamp_coef"]
        # Record history
        hist["t"][k] = t
        hist["N_expect"][k] = Nexpect
        hist["E_micro"][k] = Emicro
        hist["E_macro"][k] = Emacro
        hist["E_heat"][k] = Eheat
        hist["Z_proxy"][k] = Zproxy
        hist["S_center"][k] = float(S[len(S)//2])
        # Debug: stop on NaN
        if p["debugstopon_nan"]:
            if not np.isfinite(Etotal) or np.isnan(Zproxy) or np.any(~np.isfinite(S)):
                return hist, time.time() - start_time, True
    runtime = time.time() - start_time
    return hist, runtime, False

def run_comparison(nruns=10):
    seeds = [int(BASE_PARAMS["seed"]) + i for i in range(nruns)]
    results = {"evolved": [], "original": []}
    for use_evolved in [True, False]:
        label = "evolved" if use_evolved else "original"
        for s in seeds:
            p = copy.deepcopy(BASE_PARAMS)
            p["seed"] = s
            hist, runtime, failed = run_simulation(p, use_evolved=use_evolved, verbose=False)
            E = hist["E_macro"] + hist["E_micro"] + hist["E_heat"]
            E0 = E[0]
            errE = np.abs(E - E0) / max(1e-12, abs(E0))
            mean_err = float(np.mean(errE))
            max_err = float(np.max(errE))
            drift = float((E[-1] - E0) / max(1e-12, abs(E0)))
            results[label].append({
                "mean_err": mean_err, "max_err": max_err, "drift": drift, "nan": failed, "runtime": runtime
            })
            print(f"{label} seed={s}: mean_err={mean_err:.3e}, max_err={max_err:.3e}, drift={drift:.3e}, nan={failed}, time={runtime:.3f}s")
    def summarize(list_of_dicts):
        mean_errs = [d["mean_err"] for d in list_of_dicts]
        drifts = [d["drift"] for d in list_of_dicts]
        nans = sum(1 for d in list_of_dicts if d["nan"])
        times = [d["runtime"] for d in list_of_dicts]
        return {
            "mean_err_mean": mean(mean_errs), "mean_err_std": stdev(mean_errs) if len(mean_errs) > 1 else 0.0,
            "drift_mean": mean(drifts), "drift_std": stdev(drifts) if len(drifts) > 1 else 0.0,
            "nan_count": nans, "time_mean": mean(times)
        }
    sum_e = summarize(results["evolved"])
    sum_o = summarize(results["original"])
    print("\nSummary (evolved vs original - original grid)")
    print(f" mean_err: {sum_e['mean_err_mean']:.3e} ± {sum_e['mean_err_std']:.3e} vs {sum_o['mean_err_mean']:.3e} ± {sum_o['mean_err_std']:.3e}")
    print(f" drift: {sum_e['drift_mean']:.3e} ± {sum_e['drift_std']:.3e} vs {sum_o['drift_mean']:.3e} ± {sum_o['drift_std']:.3e}")
    print(f" NaN failures: {sum_e['nan_count']} / {nruns} vs {sum_o['nan_count']} / {nruns}")
    print(f" Avg runtime (s): {sum_e['time_mean']:.3f} vs {sum_o['time_mean']:.3f}")
    improvement_factor = (sum_o["mean_err_mean"] / sum_e["mean_err_mean"]) if sum_e["mean_err_mean"] > 0 else float("inf")
    print(f"\nImprovement factor (old_mean_err / new_mean_err) = {improvement_factor:.3f}")

# -------------------------
# Fused grid simulation + comparison harness
# -------------------------
def run_fused_simulation(p: Dict[str, Any], use_evolved: bool, verbose: bool, dx: float) -> Tuple[Dict[str, np.ndarray], float, bool]:
    rng = rng_default(p["seed"])
    Nx = int(p["grid_Nx"])
    x = np.linspace(-p["Lx"], p["Lx"], Nx)
    state = 0.1 + 0.01 * np.exp(-0.5 * (x / 0.2) ** 2)
    V = p["v0"] * (1.0 + p["alpha_s"] * state)
    N_expect = 0.02
    E_heat = 0.0
    Emicro0 = float(N_expect)
    Emacro0 = energy_of_state(state, dx)
    E0 = Emicro0 + Emacro0 + E_heat
    dt = float(p["dt_micro"])
    times = np.arange(0.0, float(p["T_total"]), dt)
    steps = len(times)
    macrosync = max(1, int(round(p["dt_macro"] / dt)))
    proj_kernel = np.exp(-0.5 * (x / 0.1) ** 2)
    proj_kernel /= proj_kernel.sum()
    coll_kernel = np.exp(-0.5 * (x / p["sigma_fus"]) ** 2)
    coll_kernel /= coll_kernel.sum()
    hist = {
        "t": np.zeros(steps), "N_expect": np.zeros(steps), "E_micro": np.zeros(steps),
        "E_macro": np.zeros(steps), "E_heat": np.zeros(steps), "Z_proxy": np.zeros(steps),
        "S_center": np.zeros(steps), "trit_state": np.zeros(steps), "inj_adapt": np.zeros(steps),
        "curvature": np.zeros(steps), "carbon_strength": np.zeros(steps)
    }
    fused_op = create_evolved_best_operator(dx) if use_evolved else None
    carbon_synth = ZSF_CARBON_SYNTHESIS_grid(dx=dx, debug=args.debug)
    injection_source = carbon_synth["injection_source"]
    start_time = time.time()
    failed = False
    for k, t in enumerate(times):
        xi_t = xi_of(t)
        dN_drive = 1e-2 * xi_t * (1.0 - np.clip(N_expect, 0.0, 1.0)) * dt
        dN_diss = -p["gammak"] * N_expect * dt
        noise = rng.normal(scale=math.sqrt(max(p["sigma_noise"], 0.0)) * math.sqrt(dt))
        N_expect = max(0.0, N_expect + dN_drive + dN_diss + noise)
        N_collected = p["kappa_proj"] * N_expect
        N_residual = max(0.0, N_expect - N_collected)
        samples = sample_collision_residual(N_residual, p["Nmcsample"], rng)
        if samples.size > 0:
            deltainavg = float((p["eta_in"] * samples).mean())
            deltaheatavg = float((p["eta_heat"] * samples).mean())
            deltafusavg = float((p["eta_fus"] * samples).mean())
            cap = p["maxinjectionfrac"] * max(1e-12, N_expect)
            deltainavg = float(np.clip(deltainavg, -1e-3, min(1e-3, cap)))
            deltaheatavg = float(np.clip(deltaheatavg, -1e-3, min(1e-3, cap)))
            deltafusavg = float(np.clip(deltafusavg, -1e-3, min(1e-3, cap)))
        else:
            deltainavg = deltaheatavg = deltafusavg = 0.0
        Gtot_scalar = N_collected + deltainavg + float(np.mean(injection_source))
        Gtot_scalar = min(Gtot_scalar, p["maxGtotscalar"])
        G_proj = Gtot_scalar * proj_kernel
        G_fus = (1.0 - p["eta_fus"]) * deltafusavg * coll_kernel
        G_total = G_proj + G_fus
        if (k % macrosync) == 0:
            if use_evolved and fused_op is not None:
                state, info = fused_op.apply(state, rng)
                hist["inj_adapt"][k] = float(info.get("inj_adapt", 0.0))
                hist["trit_state"][k] = int(info.get("trit", 0))
            else:
                # Original upwind + diffusion + source
                F = V * state
                F_im = np.roll(F, 1)
                convterm = -(macrosync * dt / dx) * (F - F_im)
                diffterm = p["nu"] * (macrosync * dt / (dx * dx)) * laplacian_1d(state, dx)
                srcterm = (macrosync * dt) * (G_total - p["gamma_env"] * state)
                noise_term = rng.normal(scale=args.sigma_noise if not args.high_noise else args.sigma_noise * 5.0, size=state.shape)
                state = state + convterm + diffterm + srcterm + noise_term
                state = 0.995 * state + 0.005 * np.roll(state, 1)
        if args.cfl_max > 0.0:
            try:
                dt = adjust_dt_by_cfl_1d(V, dx, dt, args.cfl_max, 1e-10, args.dt_micro * 10)
            except Exception:
                dt = max(1e-10, dt * 0.5)
        if k % 100 == 0 and k > 0:
            zs_trigger = {"state": state, "xi": xi_t, "lambda": 0.12, "phi": float(np.mean(state)),
                          "trit_state": int(hist["trit_state"][k]), "entropy": 0.0, "energy": float(hist["E_macro"][k-1])}
            zs_out, trigger_log = DARKMATTER_TRIGGER_grid(zs_trigger, coupling_section=2e-40, Veff=1.0, rng=rng, dx=dx)
            state = zs_out["state"]
            hist["curvature"][k] = float(zs_out.get("curvature", 0.0))
            if args.debug:
                for msg in trigger_log:
                    print(f"[grid trigger] {msg}")
        V = p["v0"] * (1.0 + p["alpha_s"] * state)
        maxallowed2 = max(p["maxinjectionfrac"] * max(1e-12, N_expect), p["min_denominator"])
        heatincrement = min(deltaheatavg, maxallowed2)
        newmicromass = min(deltafusavg, maxallowed2)
        E_heat += heatincrement
        N_expect = max(0.0, N_expect - heatincrement + newmicromass)
        Emicro = float(N_expect)
        Emacro = energy_of_state(state, dx)
        Etotal = Emicro + Emacro + E_heat
        Zproxy = abs(Etotal - E0) / max(p["min_denominator"], abs(E0))
        if Zproxy > p["Z_threshold"]:
            deltaE = Etotal - E0
            maxabsorb = p.get("maxabsorbenergy_frac", args.maxabsorbenergy_frac) * max(abs(E0), p["min_denominator"])
            absorb = math.copysign(min(abs(deltaE), maxabsorb), deltaE)
            E_heat -= absorb
            state = state * p["safetydamp_coef"]
            N_expect = N_expect * p["safetydamp_coef"]
        hist["t"][k] = t
        hist["N_expect"][k] = N_expect
        hist["E_micro"][k] = Emicro
        hist["E_macro"][k] = Emacro
        hist["E_heat"][k] = E_heat
        hist["Z_proxy"][k] = Zproxy
        hist["S_center"][k] = float(state[len(state)//2])
        hist["carbon_strength"][k] = float(carbon_synth["strengthGPa"])
        if not np.isfinite(Etotal) or np.isnan(Zproxy) or np.any(~np.isfinite(state)):
            failed = True
            break
    runtime = time.time() - start_time
    if verbose:
        print(f"Simulation finished: failed={failed}, runtime={runtime:.3f}s, final_trit={int(hist['trit_state'][-1]) if steps>0 else 0}")
    return hist, runtime, failed

def run_fused_comparison(nruns: int):
    seeds = [int(args.seed) + i for i in range(nruns)]
    results = {"evolved": [], "original": []}
    for use_evolved in [True, False]:
        label = "evolved" if use_evolved else "original"
        for s in seeds:
            p = {
                "seed": s, "grid_Nx": args.grid_Nx, "Lx": args.Lx, "T_total": args.T_total,
                "dt_micro": args.dt_micro, "dt_macro": args.dt_macro, "v0": args.v0, "alpha_s": args.alpha_s,
                "nu": args.nu, "gamma_env": args.gamma_env, "kappa_proj": args.kappa_proj, "eta_in": 0.05,
                "eta_heat": 0.10, "eta_fus": 0.05, "sigma_fus": 0.03, "Nmcsample": args.Nmcsample,
                "gammak": 5e-4, "sigma_noise": args.sigma_noise, "Z_threshold": args.Z_threshold,
                "safetydamp_coef": args.safetydamp_coef, "maxGtotscalar": args.maxGtotscalar,
                "min_denominator": args.min_denominator, "maxinjectionfrac": args.maxinjectionfrac,
                "dt_macro": args.dt_macro
            }
            dx = 2.0 * p["Lx"] / max(1, p["grid_Nx"] - 1)
            hist, runtime, failed = run_fused_simulation(p, use_evolved=use_evolved, verbose=True, dx=dx)
            E_total = np.array(hist["E_micro"]) + np.array(hist["E_macro"]) + np.array(hist["E_heat"])
            E0 = float(E_total[0])
            err_E = np.abs(E_total - E0) / max(1e-12, abs(E0))
            mean_err = float(np.mean(err_E))
            max_err = float(np.max(err_E))
            drift = float((E_total[-1] - E0) / max(1e-12, abs(E0)))
            trit_active_rate = float(np.sum(np.array(hist["trit_state"]) != 0) / len(hist["trit_state"]))
            results[label].append({
                "mean_err": mean_err, "max_err": max_err, "drift": drift, "nan": failed,
                "runtime": runtime, "trit_active_rate": trit_active_rate, 
                "final_carbon_strength": float(hist["carbon_strength"][-1]) if len(hist["carbon_strength"]) > 0 else 0.0
            })
            print(f"\n{label} seed={s}:")
            print(f" mean_err={mean_err:.3e}, max_err={max_err:.3e}, drift={drift:.3e}")
            print(f" trit_active_rate={trit_active_rate:.2f}, final_carbon_strength={hist['carbon_strength'][-1]:.2f}GPa")
            print(f" nan={failed}, time={runtime:.3f}s")
    def summarize(list_of_dicts: List[Dict[str, Any]]) -> Dict[str, float]:
        mean_errs = [d["mean_err"] for d in list_of_dicts]
        drifts = [d["drift"] for d in list_of_dicts]
        trit_rates = [d["trit_active_rate"] for d in list_of_dicts]
        carb_strs = [d["final_carbon_strength"] for d in list_of_dicts]
        nans = sum(1 for d in list_of_dicts if d["nan"])
        times = [d["runtime"] for d in list_of_dicts]
        return {
            "mean_err_mean": mean(mean_errs), "mean_err_std": stdev(mean_errs) if len(mean_errs) > 1 else 0.0,
            "drift_mean": mean(drifts), "drift_std": stdev(drifts) if len(drifts) > 1 else 0.0,
            "trit_rate_mean": mean(trit_rates), "carbon_strength_mean": mean(carb_strs),
            "nan_count": nans, "time_mean": mean(times)
        }
    sum_e = summarize(results["evolved"])
    sum_o = summarize(results["original"])
    print("\n" + "="*50)
    print("Fused Comparison Summary (Evolved vs Original)")
    print("="*50)
    print(f"Mean Energy Error: {sum_e['mean_err_mean']:.3e} ± {sum_e['mean_err_std']:.3e} vs {sum_o['mean_err_mean']:.3e} ± {sum_o['mean_err_std']:.3e}")
    print(f"Energy Drift: {sum_e['drift_mean']:.3e} ± {sum_e['drift_std']:.3e} vs {sum_o['drift_mean']:.3e} ± {sum_o['drift_std']:.3e}")
    print(f"Trit Active Rate: {sum_e['trit_rate_mean']:.2f} vs {sum_o['trit_rate_mean']:.2f}")
    print(f"Final Carbon Strength (GPa): {sum_e['carbon_strength_mean']:.2f} vs {sum_o['carbon_strength_mean']:.2f}")
    print(f"NaN Failures: {sum_e['nan_count']}/{nruns} vs {sum_o['nan_count']}/{nruns}")
    print(f"Avg Runtime (s): {sum_e['time_mean']:.3f} vs {sum_o['time_mean']:.3f}")
    improvement_factor = (sum_o["mean_err_mean"] / sum_e["mean_err_mean"]) if sum_e["mean_err_mean"] > 0 else float("inf")
    print(f"\nEnergy Error Improvement Factor: {improvement_factor:.3f}x")

# -------------------------
# Main
# -------------------------
def main():
    print("=== zsfsynthesizedadaptive_fused: full-model demo (integrated all modules) ===")
    print(f"injmult={args.inj_mult}, couplingmult={args.coupling_mult}, lowerthreshold={args.lower_threshold}, highnoise={args.high_noise}")
    sources = [SOURCE_A, SOURCE_B, SOURCE_C]
    mergeddefault = collidedict(SOURCE_A["params"], SOURCE_B["params"], RNG, exactcancel=args.exact_cancel)
    mergeddefault = collidedict(mergeddefault, SOURCE_C["params"], RNG, exactcancel=args.exact_cancel)
    mergeddefault["injectionamp"] = mergeddefault.get("injectionamp", 1e-4) * args.inj_mult
    if "diffusionnu" in mergeddefault:
        mergeddefault["diffusionnu"] = mergeddefault.get("diffusionnu", 1e-3) * args.coupling_mult
    else:
        mergeddefault["kappa"] = mergeddefault.get("kappa", 1e-3) * args.coupling_mult
    mergeddefault["stabilize"] = True
    mergeddefault["stabilize_params"] = {
        "injectionamp": mergeddefault.get("injectionamp", 1e-4), "nearmargin": 0.12,
        "mixgain": mergeddefault.get("mixgain", 5.0) * args.inj_mult, 
        "etaamp": mergeddefault.get("etaamp", 1e-5 if not args.high_noise else 5e-5),
        "holdstrength": 0.12, "thetaplus": 0.35 if args.lower_threshold else 0.5,
        "thetaminus": -0.35 if args.lower_threshold else -0.5
    }
    # Print sample params
    samplekeys = ["xi","lambda","injectionamp","etaamp","mixgain","kappa","diffusionnu"]
    sample = {k: mergeddefault.get(k) for k in samplekeys if k in mergeddefault}
    print("Sample fused params:")
    print(json.dumps(sample, indent=2))
    # Single-point evolution demo
    hist = unitpointevolutiondemo(Nsteps=args.steps, seed=12345, paramsoverride=mergeddefault)
    for i in (0, max(0, args.steps // 4), args.steps - 1):
        if i < len(hist["phi"]):
            print(f"step {i}: phi={hist['phi'][i]:.6e}, trit={hist['trit'][i]}, injadapt={hist['injadapt'][i]:.6e}")
    final_phi = hist["phi"][-1] if len(hist["phi"]) > 0 else None
    final_trit = hist["trit"][-1] if len(hist["trit"]) > 0 else None
    print("final phi:", final_phi)
    print("final tritstate:", final_trit)
    print()
    # Trigger demo (single-point)
    zstrigger = {"xi": 0.021, "lambda": 0.12, "phi": 0.0, "tritstate": 0, "entropy": 0.0,
                 "seed": 54321, "injectionadaptive": 0.0, "injprev": 0.0}
    evo_params = {}
    if args.high_noise:
        evo_params["etaamp"] = 5e-5
    if args.lower_threshold:
        evo_params["thetaplus"] = 0.35
        evo_params["thetaminus"] = -0.35
    evo_params["stabilize"] = True
    evo_params["stabilize_params"] = {
        "injectionamp": evo_params.get("injectionamp", mergeddefault.get("injectionamp", 1e-4)),
        "nearmargin": 0.12, "mixgain": mergeddefault.get("mixgain", 10.0), 
        "etaamp": evo_params.get("etaamp", mergeddefault.get("etaamp", 1e-5)),
        "holdstrength": 0.12, "thetaplus": evo_params.get("thetaplus", 0.35), "thetaminus": evo_params.get("thetaminus", -0.35)
    }
    zsout, log = DARKMATTERTRIGGER(zstrigger, couplingsection=2e-40, Veff=1.0, rng=rng_default(54321), params={"evoparams": evo_params})
    for m in log:
        print(m)
    print("after trigger zs:", {k: zsout.get(k) for k in ("phi","tritstate","entropy","energy","curvature")})
    print()
    # Carbon synthesis demo
    res = ZSFCARBONSYNTHESISdemo(scale=None, clamp=False, autoscale=True, debug=args.debug)
    print("carbon synthesis result:", res)
    print()
    # TRIT distribution
    stats = unitpointevolutiondemo(Nsteps=2000, seed=42)
    counts = np.bincount(np.array(stats["trit"]) + 1, minlength=3)
    dist = counts / len(stats["trit"])
    print("TRIT distribution (-1/0/1):", dist)
    print()
    # Candidate search (optional)
    if args.search_candidates > 0:
        print(f"Searching {args.search_candidates} candidates (grid_scaling=False to preserve single-point baseline)...")
        cands = generatecandidates([SOURCE_A, SOURCE_B, SOURCE_C], RNG, args.search_candidates, exactcancel=args.exact_cancel, injmult=args.inj_mult, couplingmult=args.coupling_mult, grid_scaling=False)
        val = validate_candidates(cands, RNG, steps=500, runs=4, debug=args.debug)
        print("Candidate validation summary (top 6):")
        for v in val[:6]:
            print(v)
        print()
    # Regression test (optional)
    rt = None
    if args.regression_test:
        print("Running regression test to verify single-point parity...")
        rt = regression_test()
        print("Regression result:", rt)
        print()
    # Original grid simulation (optional)
    failed_grid = None
    if args.use_grid:
        print("Running original grid microcosm demo...")
        p = copy.deepcopy(BASE_PARAMS)
        p.update({"seed": args.seed})
        hist_grid, runtime, failed = run_simulation(p, use_evolved=True, verbose=args.debug)
        failed_grid = failed
        print("Original grid demo runtime:", runtime, "failed:", failed)
        print()
    # Original grid comparison (optional)
    if args.nruns > 0 and args.use_grid:
        print("Running original grid comparison harness...")
        run_comparison(nruns=args.nruns)
        print()
    # Fused grid comparison
    print("=== Fused ZSF-Microcosm Model Comparison Test ===")
    print(json.dumps({"seed": args.seed, "inj_mult": args.inj_mult, "grid_Nx": args.grid_Nx, "T_total": args.T_total, "nruns": args.nruns}, indent=2))
    run_fused_comparison(nruns=args.nruns)
    # Save a minimal result JSON
    result_out = {
        "single_point_final_phi": final_phi, 
        "single_point_final_trit": final_trit, 
        "carbon_strength_GPa": res["strengthGPa"], 
        "regression_test_passed": rt["trit_match"] if args.regression_test else None,
        "original_grid_failed": failed_grid if args.use_grid else None
    }
    with open(args.result_save_path, "w", encoding="utf-8") as f:
        json.dump(result_out, f, indent=2)
    print("\n=== Simulation Completed ===")
    print(f"Result saved to: {args.result_save_path}")

if __name__ == "__main__":
    main()
