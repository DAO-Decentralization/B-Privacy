#!/usr/bin/env python3
"""
Core bribery model utilities: equilibrium solver, win probability, and
minimal budget search for swinging a proposal under public/private/noised
settings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import erf

# ——— CONFIGURATION —————————————————————————————————————————————
SIGMA: float = 1.0          # σ for each voter’s Normal(±1,σ²) preference
P_TARGET: float = 0.9       # target win probability for the target side
MAX_ITERS: int = 130         # max iterations for equilibrium Δ*
FP_TOL: float = 1e-20       # convergence tolerance for Δ*
DAMPING: float = 0.7        # Δ_new = α·F(Δ) + (1−α)·Δ
B_SEARCH_TOL: float = 1e-5  # precision for budget binary search
EPSILON: float = 1e-50      # numerical safeguard (no division by zero)
EPSILON_NOISE: float = 1    # default ε for noised advantage
MC_SAMPLES: int = 10000    # Monte Carlo trials per win‐prob call (per half; antithetic doubles)
# ——————————————————————————————————————————————————————

np.random.seed(0)

# Common random numbers (CRN) + antithetic variates cache
U_global = None
Ubar_global = None
_U_shape = (0, 0)  # (R, n)

def _ensure_U(n: int, R: int) -> None:
    """Ensure global U matrices exist with at least R rows and n columns.
    Reuses existing prefixes so results remain comparable across calls.
    """
    global U_global, Ubar_global, _U_shape
    cur_R, cur_n = _U_shape
    if U_global is None or cur_n != n:
        U_global = np.random.rand(R, n)
        Ubar_global = 1.0 - U_global
        _U_shape = (R, n)
    elif R > cur_R:
        extra = np.random.rand(R - cur_R, n)
        U_global = np.vstack([U_global, extra])
        Ubar_global = 1.0 - U_global
        _U_shape = (R, n)
    # otherwise: sufficient size, do nothing


# --- Fast math functions ---
def norm_cdf_fast(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def norm_pdf_fast(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)


# --- Data utilities ---
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["weight"] = pd.to_numeric(df.get("voting_power"), errors="coerce")
    df = df.dropna(subset=["weight", "choice"])

    # Winner per proposal by voting power
    choice_power = df.groupby(["proposal_id", "choice"])['weight'].sum().reset_index()
    winning_choices_df = choice_power.loc[choice_power.groupby('proposal_id')['weight'].idxmax()]
    winning_choice_map = winning_choices_df.set_index('proposal_id')['choice'].to_dict()

    def map_choice_to_code(row) -> int:
        winning_choice = winning_choice_map.get(row['proposal_id'])
        if winning_choice is None:
            return np.nan
        return 1 if row['choice'] == winning_choice else 2

    df['choice_code'] = df.apply(map_choice_to_code, axis=1).astype('Int64')
    df = df.dropna(subset=["choice_code"])
    return df[df.choice_code.isin([1, 2])].reset_index(drop=True)


def build_locs(choices: np.ndarray, sigma: float) -> np.ndarray:
    # code 1 (winner) maps to loc -1; code 2 maps to +1
    return np.where(choices == 1, -1.0, +1.0)


# ——— Advantage functions ————————————————————————————————————
def advantage_public(delta: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.ones_like(delta)


def advantage_private(delta: np.ndarray, w: np.ndarray) -> np.ndarray:
    return delta.copy()


def advantage_noised(delta: np.ndarray, w: np.ndarray, epsilon: float) -> np.ndarray:
    tv = 1.0 - np.exp(-epsilon * (w / 2))
    delta_final = delta + (1.0 - delta) * tv
    return delta_final

def advantage_noised_square(delta: np.ndarray, w: np.ndarray, epsilon: float) -> np.ndarray:
    width = np.log(10)/epsilon
    tv = 1 - (np.maximum(2*width - w, np.zeros_like(w))*1/(2*width))
    delta_final = delta + (1.0 - delta) * tv
    return delta_final

def b_zero(delta: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.zeros_like(delta)

# ——— Fixed-point system —————————————————————————————————————
def F_vectorized(delta: np.ndarray, w: np.ndarray, locs: np.ndarray, sigma: float,
                 adv: np.ndarray, b: np.ndarray, W: float) -> np.ndarray:
    half = W / 2.0

    delta_safe = np.maximum(delta, EPSILON)
    x = (adv / delta_safe) * b

    p = norm_cdf_fast((x - locs) / sigma)
    mu_tot = w.dot(p)
    var_tot = np.dot(w**2, p * (1 - p))

    mu_mi = mu_tot - w * p
    var_mi = var_tot - w**2 * p * (1 - p)
    sigma_mi = np.sqrt(np.clip(var_mi, EPSILON, None))

    z_yes = (mu_mi + w - half) / sigma_mi
    z_no = (mu_mi - half) / sigma_mi
    return norm_cdf_fast(z_yes) - norm_cdf_fast(z_no)


def solve_equilibrium_delta(w: np.ndarray, locs: np.ndarray, sigma: float,
                            adv_func, b_func, W: float) -> np.ndarray:
    delta = np.full_like(w, 0.5, dtype=float)
    for _ in range(MAX_ITERS):
        adv = advantage_noised(delta, w, EPSILON_NOISE) if adv_func is advantage_noised else adv_func(delta, w)
        b = b_func(delta, w)
        F_delta = F_vectorized(delta, w, locs, sigma, adv, b, W)
        delta_new = DAMPING * F_delta + (1.0 - DAMPING) * delta
        if np.max(np.abs(delta_new - delta)) < FP_TOL:
            delta = delta_new
            break
        delta = delta_new
    return delta


# ——— Win probability —————————————————————————————————————————
def mc_win_prob(w: np.ndarray, p_yes: np.ndarray, side: str,
                n_samples: int = MC_SAMPLES) -> float:
    """MC estimator with common random numbers + antithetic variates."""
    R, n = n_samples, len(w)
    _ensure_U(n, R)
    U = U_global[:R, :n]
    Ubar = Ubar_global[:R, :n]
    if side == "yes":
        hits1 = (U < p_yes)
        hits2 = (Ubar < p_yes)  # antithetic
    else:
        hits1 = (U < (1 - p_yes))
        hits2 = (Ubar < (1 - p_yes))
    totals1 = hits1.dot(w)
    totals2 = hits2.dot(w)
    W = w.sum()
    return float(0.5 * (np.mean(totals1 >= W / 2.0) + np.mean(totals2 >= W / 2.0)))


def win_prob_compute(w: np.ndarray, locs_oriented: np.ndarray, sigma: float,
                     adv: np.ndarray, delta: np.ndarray, b: np.ndarray,
                     _ignored_side: str | None = None,
                     n_samples: int = MC_SAMPLES) -> float:
    """Return Pr[target side wins] under per-voter bribes b.

    Assumes locs_oriented already encodes the target side as "yes"
    (i.e., has been flipped if necessary). This avoids double-flipping.
    """
    delta_safe = np.maximum(delta, EPSILON)
    ratio = adv / delta_safe
    p_target = norm_cdf_fast((ratio * b - locs_oriented) / sigma)
    return mc_win_prob(w, p_target, "yes", n_samples=n_samples)


# ——— Minimal budget search ———————————————————————————————————
def find_min_budget(w: np.ndarray, locs: np.ndarray, sigma: float, adv_func,
                    p_target: float, side: str,
                    initial_lo: float | None = None) -> tuple[float, np.ndarray]:
    """Find minimal budget B so that Pr[target side wins] ≥ p_target.
    Re-solves Δ* at each trial B. The `side` is the adversary target.
    """
    # Allow caller to warm-start the lower bracket with a previous solution
    lo = max(float(initial_lo), 0.0) if initial_lo is not None else 0.0
    hi = max(10000.0, 2.0 * lo)
    locs_eff = locs if side == "yes" else -locs
    W = w.sum()

    def b_allocator(delta, w_local):
        delta_safe = np.maximum(delta, EPSILON)
        adv_vec = (advantage_noised(delta, w_local, EPSILON_NOISE)
                if adv_func is advantage_noised else adv_func(delta, w_local))

        lo_lam, hi_lam = 1e-12, 1e10

        def sum_b(lam: float) -> float:
            # <- weighted objective: use w_i * adv_vec in the denominator
            arg = -2.0 * np.log(lam / (w_local * adv_vec))
            arg = np.clip(arg, 0.0, None)
            b_i = (delta_safe / adv_vec) * sigma * np.sqrt(arg)
            return float(b_i.sum())

        for _ in range(150):
            mid_lam = 0.5 * (lo_lam + hi_lam)
            if sum_b(mid_lam) > b_allocator.B_trial:
                lo_lam = mid_lam
            else:
                hi_lam = mid_lam

        lam = 0.5 * (lo_lam + hi_lam)
        arg = -2.0 * np.log(lam / (w_local * adv_vec))
        arg = np.clip(arg, 0.0, None)
        return (delta_safe / adv_vec) * sigma * np.sqrt(arg)

    def trial(B: float, R: int = MC_SAMPLES) -> tuple[float, np.ndarray]:
        """Single budget trial with adaptive MC refinement near boundary."""
        b_allocator.B_trial = B
        delta_B = solve_equilibrium_delta(w, locs_eff, sigma, adv_func, b_allocator, W)
        adv_B = advantage_noised(delta_B, w, EPSILON_NOISE) if adv_func is advantage_noised else adv_func(delta_B, w)
        b_final = b_allocator(delta_B, w)

        # Base estimate with CRN + antithetic
        p_hat = win_prob_compute(w, locs_eff, sigma, adv_B, delta_B, b_final, side,
                                 n_samples=R)

        # Adaptive refinement only when needed
        if abs(p_hat - P_TARGET) < 0.01 and R < 1_000_000:
            R2 = min(4 * R, 1_000_000)
            p_hat = win_prob_compute(w, locs_eff, sigma, adv_B, delta_B, b_final, side,
                                     n_samples=R2)

        return p_hat, b_final

    # Early exit
    win0, _ = trial(0.0)
    if win0 >= p_target:
        return 0.0, np.zeros_like(w)

    # Ensure upper bound hits target (expand from current hi)
    win_prob, b_final = trial(hi)
    while win_prob < p_target:
        hi *= 2.0
        win_prob, b_final = trial(hi)

    # Binary search by bracket width
    for _ in range(80):
        if hi - lo <= B_SEARCH_TOL:
            break
        mid = 0.5 * (lo + hi)
        win_prob, b_final = trial(mid)
        if win_prob >= p_target:
            hi = mid
        else:
            lo = mid

    return hi, b_final