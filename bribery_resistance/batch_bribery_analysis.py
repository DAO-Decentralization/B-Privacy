#!/usr/bin/env python3
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import sys
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as bribery_model

SNAPSHOT_CSV = "../data_input/all_snapshot.csv"
NOISE_RATIO_TARGET = 0.1
OUTPUT_DIR = "batch_results"

dao_name_mapping = {
    'curve.eth': 'Curve Finance',
    'uma.eth': 'UMA',
    'gnosis.eth': 'Gnosis',
    'sushigov.eth': 'Sushi',
    'bitdao.eth': 'Mantle',
    'lido-snapshot.eth': 'Lido',
    'speraxdao.eth': 'Sperax DAO',
    'cakevote.eth': 'PancakeSwap',
    'dydxgov.eth': 'dYdX',
    'beanstalkfarms.eth': 'Beanstalk Farm',
    'quickvote.eth': 'QuickSwap',
    'snapshot.dcl.eth': 'Decentraland',
    'apecoin.eth': 'Apecoin',
    'gitcoindao.eth': 'Gitcoin',
    'metislayer2.eth': 'Metis',
    'opcollective.eth': 'Optimism',
    'safe.eth': 'Safe',
    'uniswapgovernance.eth': 'Uniswap',
    'arbitrumfoundation.eth': 'Arbitrum',
    'radiantcapital.eth': 'Radiant Capital',
    'g-dao.eth': 'G DAO',
    'gmx.eth': 'GMX',
    'ens.eth': 'ENS',
    'beanstalkdao.eth': 'Beanstalk DAO',
    'aavegotchi.eth': 'Aavegotchi',
    'shellprotocol.eth': 'Shell Protocol',
    'poh.eth': 'POH'
}

def calculate_epsilon_for_target_noise(w_max: float, W_total: float) -> float:
    return (np.log(20.0)) / (NOISE_RATIO_TARGET * W_total)

def compute_proposal_bribery_costs(w: np.ndarray, locs: np.ndarray, 
                                 proposal_id: str, dao_id: str) -> dict:
    
    W_total = w.sum()
    w_max = w.max()
    
    if W_total - w_max < w_max:
        return None
    
    epsilon_target = calculate_epsilon_for_target_noise(w_max, W_total)
    
    try:
        results_yes = {}
        results_no = {}
        
        B_pub_yes, _ = bribery_model.find_min_budget(
            w, locs, bribery_model.SIGMA, bribery_model.advantage_public,
            bribery_model.P_TARGET, "yes"
        )
        B_priv_yes, _ = bribery_model.find_min_budget(
            w, locs, bribery_model.SIGMA, bribery_model.advantage_private,
            bribery_model.P_TARGET, "yes"
        )
        
        original_epsilon = bribery_model.EPSILON_NOISE
        bribery_model.EPSILON_NOISE = epsilon_target
        try:
            B_nois_yes, _ = bribery_model.find_min_budget(
                w, locs, bribery_model.SIGMA, bribery_model.advantage_noised,
                bribery_model.P_TARGET, "yes"
            )
        finally:
            bribery_model.EPSILON_NOISE = original_epsilon
        
        if not (pd.isna(B_pub_yes) or pd.isna(B_priv_yes) or pd.isna(B_nois_yes)):
            results_yes = {
                'public': B_pub_yes,
                'private': B_priv_yes, 
                'noised': B_nois_yes,
                'side': 'yes'
            }
        
        B_pub_no, _ = bribery_model.find_min_budget(
            w, locs, bribery_model.SIGMA, bribery_model.advantage_public,
            bribery_model.P_TARGET, "no"
        )
        B_priv_no, _ = bribery_model.find_min_budget(
            w, locs, bribery_model.SIGMA, bribery_model.advantage_private,
            bribery_model.P_TARGET, "no"
        )
        
        bribery_model.EPSILON_NOISE = epsilon_target
        try:
            B_nois_no, _ = bribery_model.find_min_budget(
                w, locs, bribery_model.SIGMA, bribery_model.advantage_noised,
                bribery_model.P_TARGET, "no"
            )
        finally:
            bribery_model.EPSILON_NOISE = original_epsilon
        
        if not (pd.isna(B_pub_no) or pd.isna(B_priv_no) or pd.isna(B_nois_no)):
            results_no = {
                'public': B_pub_no,
                'private': B_priv_no,
                'noised': B_nois_no,
                'side': 'no'
            }
        
        chosen_results = None
        if results_yes and results_yes['private'] > results_yes['public']:
            chosen_results = results_yes
        elif results_no and results_no['private'] > results_no['public']:
            chosen_results = results_no
        elif results_yes:
            chosen_results = results_yes
        elif results_no:
            chosen_results = results_no
        
        if chosen_results:
            public_cost = chosen_results['public']
            return {
                'dao_id': dao_id,
                'proposal_id': proposal_id,
                'public_ratio': 1.0,
                'private_ratio': chosen_results['private'] / public_cost,
                'noised_ratio_100pct': chosen_results['noised'] / public_cost,
                'target_side': chosen_results['side'],
                'epsilon_100pct': epsilon_target,
                'w_max': w_max,
                'W_total': W_total,
                'num_voters': len(w),
                'public_cost_abs': public_cost,
                'private_cost_abs': chosen_results['private'],
                'noised_cost_abs': chosen_results['noised']
            }
    
    except Exception as e:
        print(f"Error processing proposal {proposal_id[:10]}: {e}")
        return None
    
    return None

def process_single_proposal(args_tuple):
    weights, choice_codes, proposal_id, dao_id = args_tuple
    
    locs = bribery_model.build_locs(choice_codes, bribery_model.SIGMA)
    
    return compute_proposal_bribery_costs(weights, locs, proposal_id, dao_id)

def load_existing_results(output_file: str) -> pd.DataFrame:
    if os.path.exists(output_file):
        try:
            return pd.read_csv(output_file)
        except Exception as e:
            print(f"Warning: Could not load existing results from {output_file}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def save_results_batch(results_batch: list, output_file: str, force_recompute: bool = False) -> None:
    if not results_batch:
        return
    
    df_new = pd.DataFrame(results_batch)
    
    if force_recompute and os.path.exists(output_file):
        try:
            df_existing = pd.read_csv(output_file)
            for result in results_batch:
                df_existing = df_existing[~((df_existing['dao_id'] == result['dao_id']) & 
                                         (df_existing['proposal_id'] == result['proposal_id']))]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(output_file, mode='w', header=True, index=False)
        except Exception as e:
            print(f"Warning: Error handling force recompute for {output_file}: {e}")
            df_new.to_csv(output_file, mode='a', header=False, index=False)
    elif os.path.exists(output_file):
        df_new.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df_new.to_csv(output_file, mode='w', header=True, index=False)

def process_dao(dao_id: str, all_votes_df: pd.DataFrame, output_file: str, 
               num_processes: int = None, force_recompute: bool = False) -> int:
    dao_df = all_votes_df[all_votes_df["dao_id"] == dao_id].copy()
    if dao_df.empty:
        print(f"No data found for DAO: {dao_id}")
        return 0
    
    proposals = dao_df["proposal_id"].unique()
    print(f"Found {len(proposals)} proposals for {dao_id}")
    
    if force_recompute:
        proposals_to_process = proposals
        print(f"Force recompute enabled - processing all {len(proposals_to_process)} proposals for {dao_id}")
    else:
        existing_df = load_existing_results(output_file)
        if not existing_df.empty:
            existing_proposals = set(existing_df[existing_df['dao_id'] == dao_id]['proposal_id'].values)
            proposals_to_process = [p for p in proposals if p not in existing_proposals]
            print(f"Skipping {len(proposals) - len(proposals_to_process)} already computed proposals")
            print(f"Processing {len(proposals_to_process)} new proposals for {dao_id}")
        else:
            proposals_to_process = proposals
            print(f"Processing all {len(proposals_to_process)} proposals for {dao_id}")
    
    if len(proposals_to_process) == 0:
        print(f"All proposals for {dao_id} already computed")
        return 0
    
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(proposals_to_process))
    else:
        num_processes = min(num_processes, mp.cpu_count(), len(proposals_to_process))
    
    print(f"Using {num_processes} processes for parallel processing")
    
    proposal_args = []
    for pid in proposals_to_process:
        sub = dao_df[dao_df["proposal_id"] == pid]
        weights = sub["weight"].values
        choice_codes = sub["choice_code"].astype(int).values
        proposal_args.append((weights, choice_codes, pid, dao_id))
    
    processed_count = 0
    chunksize = max(1, len(proposal_args) // (num_processes * 4))
    
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_proposal, proposal_args, chunksize=chunksize),
            total=len(proposal_args),
            desc=f"Processing {dao_id}"
        ))
        
        valid_results = [r for r in results if r is not None]
        if valid_results:
            save_results_batch(valid_results, output_file, force_recompute=force_recompute)
            processed_count = len(valid_results)
    
    return processed_count

def create_geometric_mean_cost_ratio_plot(df_results: pd.DataFrame, dao_ids: list, output_file: str) -> str:
    
    dao_geometric_means = []
    
    for dao_id in df_results['dao_id'].unique():
        dao_data = df_results[df_results['dao_id'] == dao_id]
        if len(dao_data) >= 5:
            private_ratios = dao_data['private_ratio'].dropna()
            if len(private_ratios) > 0 and (private_ratios > 0).all():
                private_geometric_mean = np.exp(np.log(private_ratios).mean())
            else:
                private_geometric_mean = float('nan')
            
            noised_ratios = dao_data['noised_ratio_100pct'].dropna()
            if len(noised_ratios) > 0 and (noised_ratios > 0).all():
                noised_geometric_mean = np.exp(np.log(noised_ratios).mean())
            else:
                noised_geometric_mean = float('nan')
            
            min_swing_voters_mean = dao_data['min_swing_voters'].mean()
            
            dao_geometric_means.append({
                'dao_id': dao_id,
                'num_proposals': len(dao_data),
                'private_geometric_mean': private_geometric_mean,
                'noised_geometric_mean': noised_geometric_mean,
                'min_swing_voters_mean': min_swing_voters_mean
            })
    
    dao_geometric_df = pd.DataFrame(dao_geometric_means)
    dao_geometric_df = dao_geometric_df.sort_values('min_swing_voters_mean', ascending=False)
    
    if len(dao_geometric_df) == 0:
        print("Warning: No DAOs with 5+ proposals found for geometric mean cost ratio plot")
        return ""
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(dao_geometric_df) * 0.4)))
    
    colors = {
        'public': '#2E86C1',
        'private': '#E74C3C',
        'noised': '#28B463'
    }
    
    y_positions = np.arange(len(dao_geometric_df))
    
    ax.scatter(dao_geometric_df['private_geometric_mean'].values, y_positions, 
              color=colors['private'], s=80, alpha=0.8, label='Winner-Only', zorder=3)
    ax.scatter(dao_geometric_df['noised_geometric_mean'].values, y_positions, 
              color=colors['noised'], s=80, alpha=0.8, label='Tally Perturbation 10%', zorder=3)
    
    ax.axvline(x=1.0, color=colors['public'], linestyle='--', alpha=0.7, linewidth=2, 
               label='Public (baseline)', zorder=1)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{dao_name_mapping.get(row['dao_id'], row['dao_id'])} (n={row['num_proposals']})" for _, row in dao_geometric_df.iterrows()], fontsize=18)
    ax.set_xlabel('Relative B-Privacy (Geometric Mean)', fontsize=28)
    
    max_ratio = max(dao_geometric_df['private_geometric_mean'].max(), dao_geometric_df['noised_geometric_mean'].max())
    if max_ratio > 10:
        ax.set_xscale('log')
        ax.set_xlim(0.8, 20)
        # Create custom locator to include 1.1, 1.2, ..., 1.9
        major_ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
        minor_ticks = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x)}'))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.tick_params(axis='x', which='major', length=8, width=1.5, labelsize=16)
        ax.tick_params(axis='x', which='minor', length=4, width=1, labelsize=12)
    else:
        ax.set_xlim(0.9, max_ratio * 1.1)
        ax.xaxis.set_major_locator(plt.AutoLocator())
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=True, fancybox=True, shadow=True, fontsize=23)
    
    ax.grid(True, alpha=0.3)
    
    # Add vertical lines for all major tick values (after plot is configured)
    for tick in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]:
        ax.axvline(x=tick, color='gray', linestyle=':', alpha=0.5, linewidth=1.0, zorder=0)
    
    # Add vertical lines for 1.1, 1.2, ..., 1.9
    for i in range(1, 10):
        value = 1 + i * 0.1
        ax.axvline(x=value, color='lightgray', linestyle=':', alpha=0.4, linewidth=0.8, zorder=0)
    
    ax.text(max_ratio * 1.3, len(dao_geometric_df), 'MDC', ha='left', va='center', fontsize=14, alpha=0.8, fontweight='bold')
    for i, (_, row) in enumerate(dao_geometric_df.iterrows()):
        ax.text(max_ratio * 1.3, i, f"{row['min_swing_voters_mean']:.1f}", 
                ha='left', va='center', fontsize=16, alpha=0.7)
    
    plt.tight_layout()
    
    plot_file = output_file.replace('.csv', '_cost_ratios_geometric_mean.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return plot_file

def create_dao_summary(all_votes_df: pd.DataFrame) -> str:
    dao_stats = []
    
    for dao_id in tqdm(all_votes_df['dao_id'].unique(), desc="Computing DAO statistics"):
        dao_df = all_votes_df[all_votes_df['dao_id'] == dao_id]
        
        num_proposals = dao_df['proposal_id'].nunique()
        total_votes = len(dao_df)
        
        voters_per_proposal = dao_df.groupby('proposal_id')['voter_address'].nunique()
        avg_voters_per_proposal = voters_per_proposal.mean()
        
        voting_power_per_proposal = dao_df.groupby('proposal_id')['weight'].sum()
        avg_voting_power = voting_power_per_proposal.mean()
        
        whale_stats = []
        for pid in dao_df['proposal_id'].unique():
            proposal_votes = dao_df[dao_df['proposal_id'] == pid]
            total_power = proposal_votes['weight'].sum()
            max_power = proposal_votes['weight'].max()
            whale_pct = (max_power / total_power * 100) if total_power > 0 else 0
            whale_stats.append(whale_pct)
        
        avg_whale_percentage = np.mean(whale_stats) if whale_stats else 0
        
        dao_stats.append({
            'dao_id': dao_id,
            'num_proposals': num_proposals,
            'total_votes': total_votes,
            'avg_voters_per_proposal': avg_voters_per_proposal,
            'avg_voting_power_per_proposal': avg_voting_power,
            'avg_whale_percentage': avg_whale_percentage
        })
    
    df_summary = pd.DataFrame(dao_stats)
    df_summary = df_summary.sort_values('num_proposals', ascending=False)
    
    summary_file = f"{OUTPUT_DIR}/dao_summary.csv"
    df_summary.to_csv(summary_file, index=False)
    
    print(f"\nDAO Summary Statistics:")
    print(f"Total DAOs: {len(df_summary)}")
    print(f"Total proposals: {df_summary['num_proposals'].sum()}")
    print(f"Average proposals per DAO: {df_summary['num_proposals'].mean():.1f}")
    print(f"Average voters per proposal (across all DAOs): {df_summary['avg_voters_per_proposal'].mean():.1f}")
    
    return summary_file

def main():
    parser = argparse.ArgumentParser(description='Batch bribery analysis for multiple DAOs')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all-daos', action='store_true', help='Process all DAOs found in the input data')
    group.add_argument('--dao-ids', nargs='+', help='Specific DAO IDs to process')

    parser.add_argument('--output-file', type=str, 
                       help='Output CSV file (default: auto-generated)')
    parser.add_argument('--create-dao-summary', action='store_true',
                       help='Create DAO summary statistics file')
    parser.add_argument('--force-recompute', action='store_true',
                       help='Force recomputation of all proposals (ignore existing results)')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only generate plot from existing CSV data (no processing)')
    parser.add_argument('--num-processes', type=int, default=None,
                       help='Number of processes to use for parallel processing (default: CPU count)')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Generate output filename (consistent, no timestamp for reuse)
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = f"{OUTPUT_DIR}/bribery_costs_all_daos_normal.csv"

    # Generate plot only if requested
    if args.plot_only:
        if not os.path.exists(output_file):
            print(f"Error: CSV file not found: {output_file}")
            return 1
        
        print("Loading existing results...")
        df_results = load_existing_results(output_file)
        if df_results.empty:
            print("Error: No data found in CSV file")
            return 1
        
        all_daos_in_csv = df_results['dao_id'].unique().tolist()
        # Filter out stgdao.eth from visualization as well
        all_daos_in_csv = [dao for dao in all_daos_in_csv if (dao != 'stgdao.eth')]
        print(f"Creating visualizations for {len(all_daos_in_csv)} DAOs (excluding stgdao.eth): {', '.join(sorted(all_daos_in_csv))}")
        
        geometric_mean_plot_file = create_geometric_mean_cost_ratio_plot(df_results, all_daos_in_csv, output_file)
        print(f"Geometric mean plot saved to: {geometric_mean_plot_file}")
        return 0
        
    print("Loading and cleaning voting data...")
    if not os.path.exists(SNAPSHOT_CSV):
        print(f"Error: Data file not found: {SNAPSHOT_CSV}")
        return 1
    
    all_votes_df = bribery_model.load_and_clean(SNAPSHOT_CSV)
    print(f"Loaded {len(all_votes_df)} votes")
    
    if args.create_dao_summary:
        summary_file = create_dao_summary(all_votes_df)
        print(f"DAO summary saved to: {summary_file}")
        return 0
    
    if args.all_daos:
        dao_ids = all_votes_df['dao_id'].unique().tolist()
        dao_ids = [dao for dao in dao_ids if (dao != 'stgdao.eth')]
        print(f"\nProcessing all {len(dao_ids)} DAOs found in the dataset (excluding stgdao.eth)")
    else:
        dao_ids = args.dao_ids
        if 'stgdao.eth' in dao_ids:
            dao_ids = [dao for dao in dao_ids if (dao != 'stgdao.eth')]
            print("Note: stgdao.eth has been excluded from processing")

    total_processed = 0
    start_time = time.time()
    
    for dao_id in dao_ids:
        print(f"\n{'='*60}")
        print(f"Processing DAO: {dao_id}")
        print(f"{'='*60}")
        
        processed_count = process_dao(dao_id, all_votes_df, output_file, 
                                    num_processes=args.num_processes, 
                                    force_recompute=args.force_recompute)
        total_processed += processed_count
        
        print(f"Completed {dao_id}: {processed_count} new proposals processed")
    
    if total_processed > 0 or os.path.exists(output_file):
        df_results = load_existing_results(output_file)
        
        if not df_results.empty:
            all_daos_in_csv = df_results['dao_id'].unique().tolist()
            all_daos_in_csv = [dao for dao in all_daos_in_csv if dao != 'stgdao.eth']
            print(f"CSV contains data for {len(all_daos_in_csv)} DAOs (excluding stgdao.eth): {', '.join(sorted(all_daos_in_csv))}")
            
            df_current_run = df_results[df_results['dao_id'].isin(dao_ids)]
        
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"BATCH ANALYSIS COMPLETE")
            print(f"{'='*60}")
            print(f"Total new proposals processed: {total_processed}")
            print(f"Total proposals in current run: {len(df_current_run) if not df_current_run.empty else 0}")
            print(f"Total proposals in full dataset: {len(df_results)}")
            print(f"Total time: {elapsed:.1f} seconds")
            if total_processed > 0:
                print(f"Average time per new proposal: {elapsed/total_processed:.2f} seconds")
            print(f"Results saved to: {output_file}")
        
            if not df_current_run.empty:
                print(f"\nCost Ratio Statistics for Current Run (relative to public cost):")
                print(f"Public ratio:  mean={df_current_run['public_ratio'].mean():.4f}, std={df_current_run['public_ratio'].std():.4f}")
                print(f"Private ratio: mean={df_current_run['private_ratio'].mean():.4f}, std={df_current_run['private_ratio'].std():.4f}")
                print(f"Noised ratio:  mean={df_current_run['noised_ratio_100pct'].mean():.4f}, std={df_current_run['noised_ratio_100pct'].std():.4f}")
            
                print(f"\nSample results from current run (first 5 proposals):")
                print(df_current_run[['dao_id', 'proposal_id', 'public_ratio', 'private_ratio', 'noised_ratio_100pct']].head().to_string(index=False))
            
            print(f"\nCreating visualizations for all {len(all_daos_in_csv)} DAOs in dataset...")
            
            geometric_mean_plot_file = create_geometric_mean_cost_ratio_plot(df_results, all_daos_in_csv, output_file)
            print(f"Geometric mean plot saved to: {geometric_mean_plot_file}")
        else:
            print("No valid results found!")
            return 1
    else:
        print("No results generated!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
