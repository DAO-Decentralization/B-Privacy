import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Parse command line arguments
use_noise = '--use-noise' in sys.argv

# Read the appropriate CSV file
csv_filename = 'privacy_attack_results_noise.csv' if use_noise else 'privacy_attack_results.csv'
df = pd.read_csv(csv_filename)

# Calculate deanonymization percentages
df['voters_deanon_percentage'] = (df['total_deanonymized'] / df['total_voters']) * 100
df['voting_power_deanon_percentage'] = (df['total_voting_power_deanonymized'] / df['total_voting_power']) * 100

# Aggregate data by DAO
dao_aggregate = df.groupby('dao_id').agg({
    'voters_deanon_percentage': 'mean',
    'voting_power_deanon_percentage': 'mean', 
    'total_voters': 'mean',
    'proposal_id': 'nunique'
}).reset_index()
dao_aggregate = dao_aggregate[dao_aggregate.proposal_id >= 5]

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(11, 8))

# Use log scale for point sizes based on average number of voters
# Cap smallest points at 45 voters for better differentiation
min_voters_threshold = 45
min_size = 30
max_size = 1500

# Apply minimum threshold and calculate log scale
voters_capped = np.maximum(dao_aggregate['total_voters'], min_voters_threshold)
log_voters = np.log10(voters_capped)
log_min, log_max = log_voters.min(), log_voters.max()
normalized_sizes = (log_voters - log_min) / (log_max - log_min)
point_sizes = min_size + normalized_sizes * (max_size - min_size)

# Create scatter plot with black points
scatter = ax.scatter(
    dao_aggregate['voters_deanon_percentage'],
    dao_aggregate['voting_power_deanon_percentage'], 
    s=point_sizes,
    alpha=0.7,
    c='black',
    edgecolors='darkgray',
    linewidth=0.5
)

# Create legend with example point sizes
legend_sizes = [30, 100, 1000, 10000, 100000]  # Include <45 threshold
legend_elements = []
for size in legend_sizes:
    if size == 30:  # Special case for <45 voters
        visual_size = min_size
        label = '<45 voters'
    else:
        # Calculate the visual size for this voter count
        size_capped = max(size, min_voters_threshold)
        log_size = np.log10(size_capped)
        normalized_size = (log_size - log_min) / (log_max - log_min)
        visual_size = min_size + normalized_size * (max_size - min_size)
        label = f'{size:,} voters'
    
    legend_elements.append(
        plt.scatter([], [], s=visual_size, c='black', alpha=0.7, 
                   edgecolors='darkgray', linewidth=0.5, 
                   label=label)
    )

# Add legend with increased height to prevent overlapping
legend = ax.legend(handles=legend_elements, title='Mean Number of Voters', 
                  loc='lower right', framealpha=0.9, fontsize=18, title_fontsize=18,
                  columnspacing=2, handletextpad=1.5, labelspacing=1)

# Set labels and title with larger font sizes
ax.set_xlabel('Mean ballots leaked (%)', fontsize=25)
ax.set_ylabel('Mean voting weight leaked (%)', fontsize=25)
ax.grid(True, alpha=0.3)
# Add padding to prevent circle clipping
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
x_padding = (x_max - x_min) * 0.05  # 5% padding
y_padding = (y_max - y_min) * 0.05  # 5% padding
ax.set_xlim(x_min - x_padding, x_max + x_padding)
ax.set_ylim(0 - y_padding, y_max + y_padding)

# Increase tick label font sizes
ax.tick_params(axis='both', which='major', labelsize=25)

# Highlight and annotate the 4 specified DAOs
target_daos = ['arbitrumfoundation.eth', 'sushigov.eth', 'aavegotchi.eth', 'balancer.eth']
for i, row in dao_aggregate.iterrows():
    if row['dao_id'] in target_daos:
        dao_name = {
            'arbitrumfoundation.eth': 'Arbitrum', 
            'sushigov.eth': 'Sushi',
            'aavegotchi.eth': 'Aavegotchi', 
            'balancer.eth': 'Balancer'
        }.get(row['dao_id'])
        # Calculate the point size for this DAO
        voters_capped = max(row['total_voters'], min_voters_threshold)
        log_size = np.log10(voters_capped)
        normalized_size = (log_size - log_min) / (log_max - log_min)
        highlighted_size = min_size + normalized_size * (max_size - min_size)
        
        # Create highlighted point with red border
        ax.scatter(
            row['voters_deanon_percentage'],
            row['voting_power_deanon_percentage'],
            s=highlighted_size,
            alpha=0.9,
            c='red',
            edgecolors='darkred',
            linewidth=2,
            zorder=10
        )
        
        # Add annotation
        ax.annotate(
            dao_name, 
            (row['voters_deanon_percentage'], row['voting_power_deanon_percentage']),
            xytext=(20, 10), 
            textcoords='offset points',
            fontsize=20,
            fontweight='bold',
            alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red')
        )

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

plt.tight_layout()
plot_filename = 'plots/dao_aggregate_deanonymization_analysis_noise.png' if use_noise else 'plots/dao_aggregate_deanonymization_analysis.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print(f"DAO Aggregate Statistics ({'with noise' if use_noise else 'without noise'}):")
print(f"Using data from: {csv_filename}")
print(dao_aggregate.sort_values('total_voters', ascending=False))