#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import core as bribery_model 
from tqdm import tqdm

def calculate_epsilon_for_target_noise(w_max: float, W_total: float, noise_ratio: float) -> float:
    """Calculate epsilon such that 95% quantile of noise equals target ratio of total weight."""
    return (np.log(20.0)) / (noise_ratio * W_total)

def compute_and_plot_single_proposal(dao_id, proposal_id, noise_ratios=None):
    """
    Computes and plots the bribe distribution for a single proposal in public,
    private (noised with default epsilon), and noised (with multiple noise ratios) settings.
    
    Args:
        dao_id (str): The DAO identifier (e.g., 'apecoin.eth')
        proposal_id (str): The proposal ID to analyze
        noise_ratios (list, optional): List of noise ratios (as percentage of total weight) for noised settings.
                                     If None, uses default only.
    """
    print(f"Analyzing proposal {proposal_id[:12]}... in DAO {dao_id}")
    
    # Load and clean data
    print("Loading and cleaning data...")
    df = bribery_model.load_and_clean("../data_input/all_snapshot.csv")
    
    # Filter for the specific DAO and proposal
    dao_df = df[df['dao_id'] == dao_id].copy()
    if dao_df.empty:
        print(f"Error: No data found for DAO ID '{dao_id}'.")
        return
    
    proposal_data = dao_df[dao_df['proposal_id'] == proposal_id].copy()
    if proposal_data.empty:
        print(f"Error: Proposal ID '{proposal_id}' not found in DAO '{dao_id}'.")
        return
    
    # Extract proposal data
    w = proposal_data["weight"].values
    C = proposal_data["choice_code"].astype(int).values
    
    w_max = w.max()
    W_total = w.sum()
    
    # Check for whale dominance
    if W_total - w_max < w_max:
        print(f"Warning: Proposal has whale dominance (max weight: {w_max}, total: {W_total})")
    
    # Determine voting side (current winner)
    tally = proposal_data.groupby("choice_code")['weight'].sum()
    side = "yes" if len(tally) == 1 or tally.get(1, 0) > tally.get(2, 0) else "no"
    print(f"Voting side: {side}")
    # Choose target side as the MODEL-LOSING side at B=0 (not just observed tally)
    # Compute baseline win prob for yes/no using public mechanism and b=0
    def baseline_prob_for(side_candidate: str) -> float:
        locs_eff = bribery_model.build_locs(C, bribery_model.SIGMA) if side_candidate == "yes" else -bribery_model.build_locs(C, bribery_model.SIGMA)
        delta0 = bribery_model.solve_equilibrium_delta(w, locs_eff, bribery_model.SIGMA, bribery_model.advantage_public, bribery_model.b_zero, W_total)
        adv0 = bribery_model.advantage_public(delta0, w)
        b0 = np.zeros_like(w)
        # win_prob_compute expects the target to be treated as "yes" when locs are flipped
        return bribery_model.win_prob_compute(w, locs_eff, bribery_model.SIGMA, adv0, delta0, b0, "yes")

    p_yes_target = baseline_prob_for("yes")
    p_no_target  = baseline_prob_for("no")
    print(f"Baseline model win prob — yes: {p_yes_target:.4f}, no: {p_no_target:.4f}")

    # Target the side with smaller baseline probability
    target_side = "yes" if p_yes_target < p_no_target else "no"
    print(f"Target side (model-losing at B=0): {target_side}")
    
    # Build locations for both settings
    locs_public = bribery_model.build_locs(C, bribery_model.SIGMA)
    locs_noised = bribery_model.build_locs(C, bribery_model.SIGMA)
    
    # Compute public budget and bribes (to make target_side win)
    print("Computing public budget and bribe distribution...")
    B_pub, bribes_pub = bribery_model.find_min_budget(w, locs_public, bribery_model.SIGMA, bribery_model.advantage_public, bribery_model.P_TARGET, target_side)
    
    if pd.isna(B_pub) or B_pub < bribery_model.EPSILON:
        print(f"Error: Invalid public budget ({B_pub}) for this proposal.")
        return
    
    # Compute noised budget and bribes (with default epsilon), still targeting target_side
    print("Computing noised budget and bribe distribution (default epsilon)...")
    B_nois, bribes_nois = bribery_model.find_min_budget(w, locs_noised, bribery_model.SIGMA, bribery_model.advantage_private, bribery_model.P_TARGET, target_side)
    
    if pd.isna(B_nois):
        print(f"Error: Invalid noised budget for this proposal.")
        return
    
    # Compute noised budget and bribes with custom epsilons if provided
    B_nois_custom_list = []
    bribes_nois_custom_list = []
    budget_ratio_custom_list = []
    
    if noise_ratios is not None:
        print("Computing noised budget and bribe distribution for multiple noise ratios...")
        for noise_ratio in tqdm(noise_ratios, desc="Processing noise ratios"):
            # Calculate epsilon for target noise level
            epsilon = calculate_epsilon_for_target_noise(w_max, W_total, noise_ratio)
            print(f"noise ratio: {noise_ratio:.2%}, epsilon: {epsilon:.2e}")
            
            # Temporarily change EPSILON_NOISE
            original_epsilon = bribery_model.EPSILON_NOISE
            bribery_model.EPSILON_NOISE = epsilon
            B_nois_custom, bribes_nois_custom = bribery_model.find_min_budget(w, locs_noised, bribery_model.SIGMA, bribery_model.advantage_noised, bribery_model.P_TARGET, target_side)
            bribery_model.EPSILON_NOISE = original_epsilon
            
            if pd.isna(B_nois_custom):
                print(f"\nWarning: Invalid noised budget for epsilon = {epsilon}. Skipping this epsilon.")
                B_nois_custom_list.append(None)
                bribes_nois_custom_list.append(None)
                budget_ratio_custom_list.append(None)
            else:
                B_nois_custom_list.append(B_nois_custom)
                bribes_nois_custom_list.append(bribes_nois_custom)
                budget_ratio_custom_list.append(B_nois_custom / B_pub)
    
    # Calculate metrics
    budget_ratio = B_nois / B_pub
    noise_quantile_95 = np.log(20.0) / bribery_model.EPSILON_NOISE
    noise_ratio = noise_quantile_95 / W_total
    
    print(f"\nResults:")
    print(f"Public Budget: {B_pub:.6f}")
    print(f"Noised Budget (default ε): {B_nois:.6f}")
    print(f"Budget Ratio (default ε): {budget_ratio:.4f}")
    
    if noise_ratios is not None:
        for i, ratio in enumerate(noise_ratios):
            if B_nois_custom_list[i] is not None:
                print(f"Noised Budget (noise={ratio:.1%}): {B_nois_custom_list[i]:.6f}")
                print("Bribe Distribution:")
                print(f"Budget Ratio (noise={ratio:.1%}): {budget_ratio_custom_list[i]:.4f}")
    
    print(f"Noise Ratio: {noise_ratio:.6f}")
    
    # Print bribes with high precision
    np.set_printoptions(precision=16, suppress=False)
    # Create the plot
    plot_bribe_comparison(w, C, side, bribes_pub, bribes_nois, bribes_nois_custom_list, proposal_id, dao_id, 
                         B_pub, B_nois, B_nois_custom_list, budget_ratio, budget_ratio_custom_list, noise_ratio, noise_ratios)
    
    # Save results to CSV
    save_results_to_csv(proposal_id, dao_id, w, C, side, bribes_pub, bribes_nois, bribes_nois_custom_list, 
                       B_pub, B_nois, B_nois_custom_list, budget_ratio, budget_ratio_custom_list, noise_ratio, noise_ratios)

def plot_bribe_comparison(w, C, side, bribes_pub, bribes_nois, bribes_nois_custom_list,
                               proposal_id, dao_id, B_pub, B_nois, B_nois_custom_list,
                               budget_ratio, budget_ratio_custom_list, noise_ratio, noise_ratios):

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import ScalarFormatter, LogLocator, LogFormatterSciNotation
    import matplotlib.lines as mlines

    plt.style.use('seaborn-v0_8-whitegrid')

    # Identify winning voters
    winning_choice = 1 if side == "yes" else 2
    mask = (C == winning_choice)
    w_winning = w[mask]
    br_pub_win = bribes_pub[mask]
    br_priv_win = bribes_nois[mask]

    # Sort by weight %
    total_w = w_winning.sum()
    w_pct = (w_winning / total_w) * 100
    idx = np.argsort(w_pct)
    w_pct = w_pct[idx]
    br_pub_win = br_pub_win[idx]
    br_priv_win = br_priv_win[idx]

    # Count bribed voters
    n_pub = np.sum(br_pub_win > 0)
    n_priv = np.sum(br_priv_win > 0)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Use colorblind-friendly palette
    palette = plt.get_cmap('tab10')
    color_pub = palette(0)
    color_priv = palette(1)
    color_noise = [palette(i+2) for i in range(5)]

    # Filter out zero values for all bribe data
    # Private (noised with default epsilon)
    mask_priv = br_priv_win > 0
    w_pct_priv = w_pct[mask_priv]
    br_priv_win_filtered = br_priv_win[mask_priv]
    
    # Custom noise ratios
    w_pct_custom_list = []
    br_custom_filtered_list = []
    if noise_ratios and bribes_nois_custom_list:
        for br_cust in reversed(bribes_nois_custom_list):
            if br_cust is None:
                continue
            br_cust_win = br_cust[mask][idx]
            mask_cust = br_cust_win > 0
            w_pct_custom_list.append(w_pct[mask_cust])
            br_custom_filtered_list.append(br_cust_win[mask_cust])
    
    # Public
    mask_pub = br_pub_win > 0
    w_pct_pub = w_pct[mask_pub]
    br_pub_win_filtered = br_pub_win[mask_pub]
    
    # Plot private (noised with default epsilon) first
    if len(w_pct_priv) > 0:
        ax.plot(w_pct_priv, br_priv_win_filtered, marker='s', lw=0, markersize=10, 
                label=f"Winner-Only ({n_priv} bribed)", color=color_priv)

    # Plot custom noise ratios in descending order
    if noise_ratios and bribes_nois_custom_list:
        for i, (ratio, w_pct_cust, br_cust_filtered) in enumerate(zip(reversed(noise_ratios), w_pct_custom_list, br_custom_filtered_list)):
            if len(w_pct_cust) > 0:
                n_cust = len(w_pct_cust)
                ax.plot(w_pct_cust, br_cust_filtered, marker='^', lw=0, markersize=10,
                        color=color_noise[i % len(color_noise)],
                        label=f"Tally Perturbation {ratio:.1%} ({n_cust} bribed)")

    # Plot public last
    if len(w_pct_pub) > 0:
        ax.plot(w_pct_pub, br_pub_win_filtered, marker='o', lw=0, markersize=6, 
                label=f"Full-Disclosure ({n_pub} bribed)", color=color_pub)

    ax.set_xlabel("Weight share of voter (%)", fontsize=28)
    ax.set_ylabel("Bribe amount", fontsize=28)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Improve ticks and grid
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*.1, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*.1, numticks=12))
    formatter = LogFormatterSciNotation()
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, which='both', axis='both', alpha=0.2, linestyle='-', linewidth=0.5)

    # After plotting all series, create custom legend handles with larger markers
    legend_handles = []
    legend_handles.append(mlines.Line2D([], [], color=color_priv, marker='s', linestyle='None', markersize=14, label=f"Winner-Only ({n_priv} bribed)"))
    for i, ratio in enumerate(reversed(noise_ratios)):
        if i < len(br_custom_filtered_list):
            n_cust = len(br_custom_filtered_list[i])
            legend_handles.append(mlines.Line2D([], [], color=color_noise[i % len(color_noise)], marker='^', linestyle='None', markersize=16, label=f"Tally Perturbation {ratio:.1%} ({n_cust} bribed)"))
    legend_handles.append(mlines.Line2D([], [], color=color_pub, marker='o', linestyle='None', markersize=14, label=f"Full-Disclosure ({n_pub} bribed)"))
    legend = ax.legend(handles=legend_handles, fontsize=24, loc='lower right', frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)

    # Increase font size for axes labels and ticks
    ax.set_xlabel("Weight share of voter (%)", fontsize=28)
    ax.set_ylabel("Bribe amount", fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    # ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))  # Removed, not compatible with LogFormatterSciNotation

    plt.tight_layout()

    # Save plot instead of showing it
    output_dir = "bribe_distribution_proposal"
    os.makedirs(output_dir, exist_ok=True)
    safe_proposal_id = proposal_id.replace('0x', '')[:10]
    out_path = f"{output_dir}/bribe_comparison_{safe_proposal_id}_{dao_id.replace('.', '_')}.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def save_results_to_csv(proposal_id, dao_id, w, C, side, bribes_pub, bribes_nois, bribes_nois_custom_list, 
                       B_pub, B_nois, B_nois_custom_list, budget_ratio, budget_ratio_custom_list, noise_ratio, noise_ratios):
    """
    Saves the results to a CSV file. Only includes winning voters.
    """
    # Determine which voters voted for the winning side
    winning_choice = 1 if side == "yes" else 2
    winning_voter_mask = (C == winning_choice)
    
    # Filter data for winning voters only
    w_winning = w[winning_voter_mask]
    bribes_pub_winning = bribes_pub[winning_voter_mask]
    bribes_nois_winning = bribes_nois[winning_voter_mask]
    C_winning = C[winning_voter_mask]
    
    # Create results dataframe
    results_data = []
    for i in range(len(w_winning)):
        row_data = {
            'proposal_id': proposal_id,
            'dao_id': dao_id,
            'weight': w_winning[i],
            'choice_code': C_winning[i],
            'voting_side': side,
            'public_bribe': bribes_pub_winning[i],
            'private_bribe_default': bribes_nois_winning[i],
            'bribe_ratio_default': bribes_nois_winning[i] / bribes_pub_winning[i] if bribes_pub_winning[i] > 0 else np.nan
        }
        
        # Add custom noise ratio results
        if noise_ratios is not None:
            for j, ratio in enumerate(noise_ratios):
                if bribes_nois_custom_list[j] is not None:
                    row_data[f'private_bribe_noise{ratio:.2%}'] = bribes_nois_custom_list[j][i]
                    row_data[f'bribe_ratio_noise{ratio:.2%}'] = bribes_nois_custom_list[j][i] / bribes_pub[i] if bribes_pub[i] > 0 else np.nan
        
        results_data.append(row_data)
    
    results_df = pd.DataFrame(results_data)
    
    # Add summary row
    summary_data = {
        'proposal_id': proposal_id,
        'dao_id': dao_id,
        'weight': 'SUMMARY',
        'public_bribe': B_pub,
        'private_bribe_default': B_nois,
        'bribe_ratio_default': budget_ratio
    }
    
    # Add custom noise ratio summary results
    if noise_ratios is not None:
        for j, ratio in enumerate(noise_ratios):
            if B_nois_custom_list[j] is not None:
                summary_data[f'private_bribe_noise{ratio:.2%}'] = B_nois_custom_list[j]
                summary_data[f'bribe_ratio_noise{ratio:.2%}'] = budget_ratio_custom_list[j]
    
    summary_row = pd.DataFrame([summary_data])
    results_df = pd.concat([results_df, summary_row], ignore_index=True)
    
    # Save to CSV
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    safe_proposal_id = proposal_id.replace('0x', '')[:10]
    output_file = f'{output_dir}/bribe_comparison_{safe_proposal_id}_{dao_id.replace(".", "_")}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    """
    Main function to compute and plot bribe distribution for a single proposal.
    """
    parser = argparse.ArgumentParser(description="Compute and plot bribe distribution for a single proposal in public and private settings.")
    parser.add_argument("dao_id", type=str, help="The DAO identifier (e.g., 'apecoin.eth')")
    parser.add_argument("proposal_id", type=str, help="The proposal ID to analyze")
    
    args = parser.parse_args()
    
    # Default noise ratios array if none provided (as percentages of total weight)
    noise_ratios = [0.01, 0.1, 0.5]  # 1%, 5%, 10%, 25%, 50% of total weight
    compute_and_plot_single_proposal(args.dao_id, args.proposal_id, noise_ratios)

if __name__ == "__main__":
    main() 