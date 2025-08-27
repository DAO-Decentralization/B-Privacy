#!/usr/bin/env python3
"""
Single-step script: Given a DAO name, compute bribery cost vs. noise,
enrich the data with clustering metrics, save the CSV, and plot the
clustered average curve. Only argument required is the DAO name.

Now includes incremental data saving to a single CSV file with duplicate checking.

UPDATE: Parallelized across proposals using ProcessPoolExecutor (max workers = CPU cores).
"""

import os
import time
import json
import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import core as bribery_model

NUM_NOISE95_POINTS = 5  # number of points to later interpolate the curve
NOISE95_RANGE = (1e-2, 1e0)  # default range for 95% quantile ratios (min, max) as fraction of total weight
OUTPUT_DIR = "bribery_cost_against_noise"
SNAPSHOT_CSV = "../data_input/all_snapshot.csv"
PROPOSAL_TIMEOUT_SECONDS = 500  # 5 minutes
CHECKPOINT_INTERVAL = 1  # Save checkpoint every N completed proposals
 
def safe_id(dao_id: str) -> str:
    return "".join(c for c in dao_id if c.isalnum() or c in (".", "_")).rstrip()


def get_noised_budget_for_epsilon(
    w: np.ndarray,
    locs: np.ndarray,
    sigma: float,
    p_target: float,
    side: str,
    epsilon: float,
    initial_lo: float | None = None,
) -> Tuple[float, np.ndarray]:
    """Call find_min_budget with a temporary EPSILON_NOISE value."""
    original_epsilon = bribery_model.EPSILON_NOISE
    bribery_model.EPSILON_NOISE = epsilon
    try:
        budget, bribes = bribery_model.find_min_budget(
            w, locs, sigma, bribery_model.advantage_noised, p_target, side,
            initial_lo=initial_lo
        )
    finally:
        bribery_model.EPSILON_NOISE = original_epsilon
    return budget, bribes


def save_checkpoint(results: List[dict], dao_id: str, output_dir: str, checkpoint_num: int) -> str:
    """Save a checkpoint of the current results."""
    if not results:
        return ""
    
    safe_dao = safe_id(dao_id)
    checkpoint_file = os.path.join(output_dir, f"computed_data_{safe_dao}_small.csv")
    pd.DataFrame(results).to_csv(checkpoint_file, index=False)
    print(f"  Checkpoint {checkpoint_num} saved: {len(results)} results -> {checkpoint_file}")
    return checkpoint_file


def load_existing_results(dao_id: str, output_dir: str) -> Tuple[List[dict], set]:
    """Load existing results from the main CSV file and return processed proposal IDs."""
    safe_dao = safe_id(dao_id)
    csv_file = os.path.join(output_dir, f"computed_data_{safe_dao}_small.csv")
    
    if not os.path.exists(csv_file):
        print(f"No existing results file found: {csv_file}")
        return [], set()
    
    try:
        df_existing = pd.read_csv(csv_file)
        if df_existing.empty:
            print(f"Existing results file is empty: {csv_file}")
            return [], set()
        processed_proposals = set(df_existing['proposal_id'].unique())
        print(f"Found existing results file: {csv_file}")
        print(f"Already processed {len(processed_proposals)} proposals")
        return df_existing.to_dict('records'), processed_proposals
    except Exception as e:
        print(f"Error reading existing results file: {e}")
        return [], set()

def _process_single_proposal(
    pid: str,
    w: np.ndarray,
    C: np.ndarray,
) -> List[dict]:
    """
    Compute all noise points for a single proposal (serially).
    Returns a list of rows (dicts) for that proposal; empty list if skipped.
    """
    # Build locs and quick feasibility checks
    locs = bribery_model.build_locs(C, bribery_model.SIGMA)

    w_max = float(w.max())
    W_total = float(w.sum())

    if W_total - w_max < w_max:
        return []

    B_pub_yes, _ = bribery_model.find_min_budget(
        w, locs, bribery_model.SIGMA, bribery_model.advantage_public,
        bribery_model.P_TARGET, "yes"
    )
    B_priv_yes, br_priv_yes = bribery_model.find_min_budget(
        w, locs, bribery_model.SIGMA, bribery_model.advantage_private,
        bribery_model.P_TARGET, "yes"
    )
    B_pub_no, _ = bribery_model.find_min_budget(
        w, locs, bribery_model.SIGMA, bribery_model.advantage_public,
        bribery_model.P_TARGET, "no"
    )
    B_priv_no, br_priv_no = bribery_model.find_min_budget(
        w, locs, bribery_model.SIGMA, bribery_model.advantage_private,
        bribery_model.P_TARGET, "no"
    )

    if B_priv_yes > B_pub_yes and B_priv_no <= B_pub_no:
        side = "yes"
    elif B_priv_no > B_pub_no and B_priv_yes <= B_pub_yes:
        side = "no"
    else:
        ratio_yes = B_priv_yes / B_pub_yes
        ratio_no = B_priv_no / B_pub_no
        side = "yes" if ratio_yes > ratio_no else "no"

    B_pub = B_pub_yes if side == "yes" else B_pub_no
    if (B_pub is None) or (not np.isfinite(B_pub)) or (B_pub < bribery_model.EPSILON):
        return []

    B_priv = B_priv_yes if side == "yes" else B_priv_no
    private_ratio = (B_priv / B_pub) if (pd.notna(B_priv) and B_pub > 0) else np.nan

    noise95_grid = np.logspace(np.log10(NOISE95_RANGE[0]), np.log10(NOISE95_RANGE[1]), NUM_NOISE95_POINTS)
    start_time = time.time()
    prev_b_nois: float | None = None
    rows: List[dict] = []

    for noise_ratio in noise95_grid:
        if time.time() - start_time > PROPOSAL_TIMEOUT_SECONDS:
            break

        eps = (np.log(10.0)) / (noise_ratio * W_total)
        b_nois, bribes_nois = get_noised_budget_for_epsilon(
            w, locs, bribery_model.SIGMA, bribery_model.P_TARGET, side, eps, initial_lo=prev_b_nois
        )

        if pd.notna(b_nois) and b_nois > bribery_model.EPSILON and B_pub > bribery_model.EPSILON:
            if prev_b_nois is not None and b_nois < prev_b_nois:
                b_nois = prev_b_nois
            budget_ratio = b_nois / B_pub
            prev_b_nois = b_nois

            bribe_dist_json = json.dumps(list(zip(w.tolist(), bribes_nois.tolist())))
            rows.append(
                {
                    "proposal_id": pid,
                    "noise_ratio": float(noise_ratio),
                    "budget_ratio": float(budget_ratio),
                    "bribe_distribution": bribe_dist_json,
                    "private_ratio": float(private_ratio) if pd.notna(private_ratio) else np.nan,
                }
            )
    return rows


def compute_results_for_dao(df: pd.DataFrame, target_dao_id: str, output_dir: str) -> pd.DataFrame:
    """Compute bribery cost vs. noise results for all proposals in the DAO with incremental saving."""
    dao_df = df[df["dao_id"] == target_dao_id].copy()
    if dao_df.empty:
        return pd.DataFrame()

    all_results, processed_proposals = load_existing_results(target_dao_id, output_dir)
    
    proposals_all = dao_df["proposal_id"].unique().tolist()
    if processed_proposals:
        proposals_to_process = [pid for pid in proposals_all if pid not in processed_proposals]
        print(f"Resuming computation: {len(processed_proposals)} proposals already processed, {len(proposals_to_process)} remaining")
    else:
        proposals_to_process = proposals_all
        print(f"Starting fresh computation for {target_dao_id}")

    print(f"Found {len(proposals_to_process)} proposals to process for {target_dao_id}.")

    if not proposals_to_process:
        return pd.DataFrame(all_results)

    payloads = []
    for pid in proposals_to_process:
        sub = dao_df[dao_df["proposal_id"] == pid]
        w = sub["weight"].values.astype(float)
        C = sub["choice_code"].astype(int).values
        payloads.append((pid, w, C))

    max_workers = max(1, os.cpu_count() or 1)
    completed = 0
    checkpoint_num = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_proposal, pid, w, C): pid for (pid, w, C) in payloads}

        with tqdm(total=len(futures), desc=f"Processing {target_dao_id} proposals (parallel)") as pbar:
            for fut in as_completed(futures):
                pid = futures[fut]
                try:
                    proposal_results = fut.result()
                except Exception as e:
                    print(f"  Error in proposal {pid[:10]}: {e}")
                    proposal_results = []

                all_results.extend(proposal_results)
                completed += 1
                pbar.update(1)

                if completed % CHECKPOINT_INTERVAL == 0:
                    checkpoint_num += 1
                    save_checkpoint(all_results, target_dao_id, output_dir, checkpoint_num)
                    print(f"  Completed proposal {pid[:10]}: {len(proposal_results)} noise points")

    if all_results:
        save_checkpoint(all_results, target_dao_id, output_dir, 0)
    
    return pd.DataFrame(all_results)


def compute_weight_entropy(weights):
    """
    Compute the entropy of voting weights for a proposal.
    """
    total_weight = np.sum(weights)
    if total_weight == 0:
        return 0.0
    probabilities = weights / total_weight
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy


def compute_min_swing_voters(proposal_votes):
    """
    Compute the minimum number of voters needed to swing the election.
    """
    if len(proposal_votes) == 0:
        return 0
    choice_totals = proposal_votes.groupby('choice_code')['weight'].sum().sort_values(ascending=False)
    if len(choice_totals) < 2:
        return 0
    winning_choice = choice_totals.index[0]
    winning_total = choice_totals.iloc[0]
    second_total = choice_totals.iloc[1]
    margin = winning_total - second_total
    if margin <= 0:
        return 0
    winning_voters = proposal_votes[proposal_votes['choice_code'] == winning_choice].copy()
    winning_voters = winning_voters.sort_values('weight', ascending=False).reset_index(drop=True)
    cumulative_weight = 0
    voters_needed = 0
    for _, row in winning_voters.iterrows():
        cumulative_weight += row['weight']
        voters_needed += 1
        new_winning_total = winning_total - cumulative_weight
        new_second_total = second_total + cumulative_weight
        if new_winning_total <= new_second_total:
            return voters_needed
    return len(winning_voters)


def compute_additional_metrics(df_all_votes: pd.DataFrame, dao_id: str, proposal_ids: np.ndarray) -> pd.DataFrame:
    """Compute per-proposal metrics needed for clustering and context."""
    dao_votes_df = df_all_votes[df_all_votes["dao_id"] == dao_id].copy()
    new_rows = []
    for pid in tqdm(proposal_ids, desc=f"Computing metrics for {dao_id}"):
        proposal_votes = dao_votes_df[dao_votes_df["proposal_id"] == pid]
        if proposal_votes.empty:
            continue

        total_voting_power = proposal_votes["weight"].sum()
        total_voters = proposal_votes["voter_address"].nunique()

        weights = proposal_votes["weight"].values
        choice_codes = proposal_votes["choice_code"].astype(int).values
        
        proposal_df = pd.DataFrame({'choice_code': choice_codes, 'weight': weights})
        min_swing_voters = compute_min_swing_voters(proposal_df)
        weight_entropy = compute_weight_entropy(np.array(weights))

        sorted_weights = proposal_votes.sort_values("weight", ascending=False)["weight"]
        cumulative_power = 0.0
        voters_count = 0
        for weight in sorted_weights:
            cumulative_power += weight
            voters_count += 1
            if cumulative_power > total_voting_power / 2.0:
                break

        whale_power = proposal_votes["weight"].max()
        whale_percentage = (whale_power / total_voting_power) * 100.0 if total_voting_power > 0 else 0.0

        winning_votes = proposal_votes[proposal_votes["choice_code"] == 1]
        winning_power = winning_votes["weight"].sum()
        winning_voters = winning_votes["voter_address"].nunique()

        winning_power_percentage = (winning_power / total_voting_power) * 100.0 if total_voting_power > 0 else 0.0
        winning_voter_percentage = (winning_voters / total_voters) * 100.0 if total_voters > 0 else 0.0

        new_rows.append(
            {
                "proposal_id": pid,
                "whale_power_percentage": whale_percentage,
                "winning_option_power_percentage": winning_power_percentage,
                "winning_option_voter_percentage": winning_voter_percentage,
                "voters_for_50_percent_power": voters_count,
                "min_swing_voters": min_swing_voters,
                "weight_entropy": weight_entropy,
            }
        )

    return pd.DataFrame(new_rows)


def plot_clustered_average(df: pd.DataFrame, target_dao_id: str, output_dir: str) -> None:
    """Create and save the clustered average plot from an already enriched DataFrame."""
    if df.empty:
        print("The data is empty. No plot will be generated.")
        return
    if "min_swing_voters" not in df.columns:
        print("Error: 'min_swing_voters' column not found in provided data.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    unique_voter_counts = sorted(df.groupby("proposal_id")["min_swing_voters"].first().unique())
    print(f"Unique swing voter counts: {unique_voter_counts}")
    
    if len(unique_voter_counts) == 1:
        vmin = unique_voter_counts[0] * 0.9
        vmax = unique_voter_counts[0] * 1.1
    else:
        vmin = min(unique_voter_counts)
        vmax = max(unique_voter_counts)
    
    _ = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cohort_colors = ['#0072B2', '#009E73', '#E69F00', '#CC79A7']

    print(f"Found {len(unique_voter_counts)} unique swing voter counts")

    mdc_ranges = [
        (1, 10),
        (10, 20),
        (20, 30),
        (30, float('inf'))
    ]
    
    clusters: List[Tuple[List[int], int]] = []
    for mdc_min, mdc_max in mdc_ranges:
        cluster_proposals = df.groupby("proposal_id")["min_swing_voters"].first()
        if mdc_max == float('inf'):
            cluster_proposals = cluster_proposals[cluster_proposals >= mdc_min]
        else:
            cluster_proposals = cluster_proposals[(cluster_proposals >= mdc_min) & (cluster_proposals < mdc_max)]
        clusters.append((list(cluster_proposals.values), len(cluster_proposals)))

    for cluster_idx, (voter_counts_in_cluster, total_proposals) in enumerate(clusters):
        if total_proposals == 0:
            continue
        proposals_in_cluster = df.groupby("proposal_id")["min_swing_voters"].first()
        cluster_proposals = proposals_in_cluster[proposals_in_cluster.isin(voter_counts_in_cluster)].index

        if len(cluster_proposals) == 0:
            continue

        cluster_data = df[df["proposal_id"].isin(cluster_proposals)]
        all_noise_ratios = np.sort(cluster_data["noise_ratio"].unique())
        all_noise_ratios = all_noise_ratios[(all_noise_ratios >= 1e-2) & (all_noise_ratios <= 1e0)]

        interpolated_budgets: List[np.ndarray] = []
        valid_proposals_in_cluster = []
        for pid in cluster_proposals:
            proposal_data = cluster_data[cluster_data["proposal_id"] == pid].sort_values("noise_ratio")
            if len(proposal_data) >= 5:
                from scipy.interpolate import interp1d
                try:
                    f = interp1d(
                        proposal_data["noise_ratio"],
                        proposal_data["budget_ratio"],
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    interpolated_budgets.append(f(all_noise_ratios))
                    valid_proposals_in_cluster.append(pid)
                except Exception:
                    continue

        if len(interpolated_budgets) == 0:
            continue

        avg_budgets = np.mean(interpolated_budgets, axis=0)
        color = cohort_colors[cluster_idx % len(cohort_colors)]
        ax.plot(all_noise_ratios, avg_budgets, linestyle="-", color=color, linewidth=3, alpha=1.0)

        min_voters = min(voter_counts_in_cluster)
        if min_voters < 10:
            label_text = f"1-10 MDC (n={{}})".format(len(valid_proposals_in_cluster))
        elif min_voters < 20:
            label_text = f"10-20 MDC (n={{}})".format(len(valid_proposals_in_cluster))
        elif min_voters < 30:
            label_text = f"20-30 MDC (n={{}})".format(len(valid_proposals_in_cluster))
        else:
            label_text = f"30+ MDC (n={{}})".format(len(valid_proposals_in_cluster))

        mid_x_idx = len(all_noise_ratios) // 2
        ax.text(
            all_noise_ratios[mid_x_idx],
            avg_budgets[mid_x_idx],
            label_text,
            fontsize=20,
            fontweight='bold',
            color=color,
            ha='center',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color),
            zorder=10
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-2, 1e0)
    
    # Set y-axis limits with more space above the top line
    y_min = 1e0  # Start at 1
    y_max = 3e5  # End at 200,000 (more space above the ~80,000 top line)
    ax.set_ylim(y_min, y_max)
    
    x_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    x_tick_labels = ['1%', '2%', '5%', '10%', '20%', '50%', '100%']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel("Noise Ratio", fontsize=28)
    ax.set_ylabel("Relative B-Privacy", fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, axis='y', which='major', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(False, axis='x', which='both')
    ax.grid(False, axis='y', which='minor')
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    safe_dao = safe_id(target_dao_id)
    plot_filename = os.path.join(output_dir, f"plot_clustered_average_{safe_dao}.png")
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()
    print(f"\nClustered average plot for {target_dao_id} saved to {plot_filename}")


def run_pipeline(dao_id: str) -> None:
    print(f"Starting end-to-end analysis for DAO: {dao_id}")
    print("Loading and cleaning data...")
    all_votes_df = bribery_model.load_and_clean(SNAPSHOT_CSV)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Compute results with incremental saving (parallel across proposals)
    results_df = compute_results_for_dao(all_votes_df, dao_id, OUTPUT_DIR)
    if results_df.empty:
        print(f"No valid results were computed for {dao_id}.")
        return

    # 2) Enrich with per-proposal metrics (only if they don't already exist)
    if 'min_swing_voters' not in results_df.columns or 'weight_entropy' not in results_df.columns:
        print("Computing additional metrics...")
        proposal_ids = results_df["proposal_id"].unique()
        metrics_df = compute_additional_metrics(all_votes_df, dao_id, proposal_ids)
        enriched_df = pd.merge(results_df, metrics_df, on="proposal_id", how="left")
    else:
        print("Additional metrics already exist, skipping computation...")
        enriched_df = results_df

    # 3) Save final CSV
    safe_dao = safe_id(dao_id)
    csv_path = os.path.join(OUTPUT_DIR, f"computed_data_{safe_dao}_small.csv")
    enriched_df.to_csv(csv_path, index=False)
    print(f"Saved final enriched results to {csv_path}")

    # 4) Plot
    plot_clustered_average(enriched_df, dao_id, OUTPUT_DIR)


def main() -> None:
    global NOISE95_RANGE, NUM_NOISE95_POINTS
    parser = argparse.ArgumentParser(
        description="Compute, save, and plot bribery resistance metrics for a DAO (choose 95% noise quantile range)"
    )
    parser.add_argument("dao_id", type=str, help="DAO identifier, e.g. 'apecoin.eth'")
    parser.add_argument(
        "--noise95-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=list(NOISE95_RANGE),
        help="Range for 95%% quantile of noise as a fraction of total weight",
    )
    parser.add_argument(
        "--noise95-points",
        type=int,
        default=NUM_NOISE95_POINTS,
        help="Number of grid points for the 95%% quantile noise range",
    )
    args = parser.parse_args()
    
    NOISE95_RANGE = (float(args.noise95_range[0]), float(args.noise95_range[1]))
    NUM_NOISE95_POINTS = int(args.noise95_points)
    run_pipeline(args.dao_id)


if __name__ == "__main__":
    main()
