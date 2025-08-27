## Overview

The DAO Voting Privacy Repository contains the artifacts from the paper _B-Privacy: Defining and Enforcing Privacy in Weighted Voting_. B-Privacy is a metric that captures the economic cost to an adversary of bribing voters based on revealed voting tallies. We propose a mechanism to boost B-privacy by noising voting tallies. The main components of the repository are scripts to analyze bribery resistance in DAO snapshot voting. The workflow separates heavy computations (producing CSVs) from fast visualization. The repository contains the main sections:
- `/attack`: Related to *Section 4, Practical Attacks on Raw Tallies*
- `/bribery_resistance`: Related to *Section 8, Empirical Analysis of B-Privacy*

## Environment

- Python 3.10+
- Optional:
```bash
python3 -m venv venv
source venv/bin/activate
```
- Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Data Setup
Data input is expected at `data_input/all_snapshot.csv`. To unzip the file `data_input/all_snapshot.zip`, follow the below instructions:
- Install Git LFS: https://git-lfs.com/
```bash
brew install git-lfs
```
- At the base of the repo, run:
```bash
git lfs install
git lfs pull
```

## Attacks on Tally data

Generates a csv that shows data on the success of revealing voter choices based on just the end tally result for a DAO proposal. Used to reproduce the following:
- _Section 4.2, Figure 1: Unified attack algorithm results across all DAOs_, 
- _Section 4.2, Figure 2: Attack effectiveness breakdown for four DAOs spanning different scales_, and
- _Section 7.1, Figure 4: Comparison of raw tally versus adapted noised tally attack effectiveness across all DAOs_.

Run from the `attack/` directory:

### Process attack on all DAOs
_Expected runtime: 30 mins_
```bash
g++ guess_voters.cc -o guess_voters
./guess_voters
```

### Get plots for attack effectiveness
```bash
python3 plot_dao_aggregate_analysis.py
```

```bash
python3 plot_dao_analysis.py
```

Outputs:
- CSV: `attack/privacy_attack_results.csv`
- Summary Plot: `attack/plots/dao_aggregate_deanonymization_analysis.png`
- DAO-specific Plots: `attack/plots/dao_deanonymization_analysis.png`

## Batch bribery analysis

Generates bribery cost ratios for public, private, and noised mechanisms and saves results to `bribery_resistance/batch_results/bribery_costs_all_daos_normal.csv`. Also produces a scatter plot of average ratios per DAO. Used to reproduce the following:
- _Section 8.2, Figure 5: Relative B-Privacy by DAO under different Tally algorithms_. 

Run from the `bribery_resistance/` directory:

### Process all DAOs
_Expected runtime: 2+ hours_
```bash
python3 batch_bribery_analysis.py --all-daos
```

### Process specific DAOs
_Expected runtime: 30 mins_
```bash
python3 batch_bribery_analysis.py --dao-ids apecoin.eth balancer.eth
```

### Plot only (uses existing CSV output from above)
```bash
python3 batch_bribery_analysis.py --plot-only
```

### Useful flags
- `--num-processes N`: limit parallel workers
- `--force-recompute`: ignore existing rows and recompute
- `--create-dao-summary`: writes `dao_summary.csv` with simple stats
- `--output-file PATH`: override output CSV path

Outputs:
- CSV: `bribery_resistance/batch_results/bribery_costs_all_daos_normal.csv`
- Plot: `bribery_resistance/batch_results/bribery_costs_all_daos_normal_cost_ratios.png`
- Optional summary: `bribery_resistance/batch_results/dao_summary.csv`

## Bribery cost vs noise (per DAO)

Computes, incrementally saves, and plots how required bribery scales with noise across proposals of a DAO. Results are saved under `bribery_resistance/bribery_cost_against_noise/`. Used to reproduce the following:
- _Section 8.3, Figure 6: B-Privacy as a function of the tally perturbation d, grouped by minimum decisive coalition (MDC) size_.

_Expected runtime: 45 mins_
```bash
python3 bribery_cost_against_noise.py apecoin.eth
```

Outputs:
- CSV: `bribery_resistance/bribery_cost_against_noise/computed_data_<dao>_small.csv`
- Plot: `bribery_resistance/bribery_cost_against_noise/plot_clustered_average_<dao>.png`

Optional arguments:
- `--noise95-range LOW HIGH` (default: 1e-3 1e0)
- `--noise95-points N` (default: 50)

## Bribe distribution for a single proposal

Computes and compares bribe distributions for a proposal under public vs private mechanisms, and optional additional noise ratios. Saves a figure and a CSV with per-voter data. Used to reproduce the following:
- _Section 8.3, Figure 7: Optimal bribe distribution across voters for the winner, corrected noised, and full-disclosure tally algorithms_.

_Expected runtime: 15 min_
```bash
python3 bribe_distribution_proposal.py apecoin.eth 0x<proposal_id>
```

Outputs:
- Plot: `bribe_distribution_proposal/bribe_comparison_<shortid>_<dao>.png`
- CSV: `results/bribe_comparison_<shortid>_<dao>.csv`

## Quick start

```bash
cd bribery_resistance
python3 batch_bribery_analysis.py --all-daos
python3 batch_bribery_analysis.py --plot-only
python3 bribery_cost_against_noise.py apecoin.eth
python3 bribe_distribution_proposal.py apecoin.eth 0x<proposal_id>
```