import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read both CSV files
df_normal = pd.read_csv('privacy_attack_results.csv')
df_noise = pd.read_csv('privacy_attack_results_noise.csv')

# Calculate deanonymization percentages for both datasets
# For normal data, use total deanonymized (all guesses are correct without noise)
df_normal['voters_deanon_percentage'] = (df_normal['total_deanonymized'] / df_normal['total_voters']) * 100
df_normal['voting_power_deanon_percentage'] = (df_normal['total_voting_power_deanonymized'] / df_normal['total_voting_power']) * 100

# For noise data, use correct counts/power (since noise can cause incorrect guesses)
df_noise['voters_deanon_percentage'] = (df_noise['total_correct'] / df_noise['total_voters']) * 100
df_noise['voting_power_deanon_percentage'] = (df_noise['total_voting_power_correct'] / df_noise['total_voting_power']) * 100

# Aggregate data by DAO for both datasets
dao_normal = df_normal.groupby('dao_id').agg({
    'voters_deanon_percentage': 'mean',
    'voting_power_deanon_percentage': 'mean', 
    'total_voters': 'mean',
    'proposal_id': 'nunique'
}).reset_index()
dao_normal = dao_normal[dao_normal.proposal_id >= 5]

dao_noise = df_noise.groupby('dao_id').agg({
    'voters_deanon_percentage': 'mean',
    'voting_power_deanon_percentage': 'mean', 
    'total_voters': 'mean',
    'proposal_id': 'nunique'
}).reset_index()
dao_noise = dao_noise[dao_noise.proposal_id >= 5]

# Merge the datasets to compare same DAOs
comparison = dao_normal.merge(dao_noise, on='dao_id', suffixes=('_normal', '_noise'))

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(12, 9))

# Use log scale for point sizes based on average number of voters
min_voters_threshold = 45
min_size = 30
max_size = 1500

# Apply minimum threshold and calculate log scale
voters_capped = np.maximum(comparison['total_voters_normal'], min_voters_threshold)
log_voters = np.log10(voters_capped)
log_min, log_max = log_voters.min(), log_voters.max()
normalized_sizes = (log_voters - log_min) / (log_max - log_min)
point_sizes = min_size + normalized_sizes * (max_size - min_size)

# Plot original positions (without noise) as ghost points
ax.scatter(
    comparison['voters_deanon_percentage_normal'],
    comparison['voting_power_deanon_percentage_normal'],
    s=point_sizes,
    alpha=0.3,
    c='lightgray',
    edgecolors='none',
    linewidth=0,
    label='Raw tally'
)

# Plot final positions (with noise) as prominent points
ax.scatter(
    comparison['voters_deanon_percentage_noise'],
    comparison['voting_power_deanon_percentage_noise'],
    s=point_sizes,
    alpha=0.7,
    c='black',
    edgecolors='darkgray',
    linewidth=0.5,
    label='Noised tally'
)

# Draw arrows showing the change
for i, row in comparison.iterrows():
    x_start = row['voters_deanon_percentage_normal']
    y_start = row['voting_power_deanon_percentage_normal']
    x_end = row['voters_deanon_percentage_noise']
    y_end = row['voting_power_deanon_percentage_noise']
    
    # Calculate arrow length for color coding
    arrow_length = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    
    # Only draw arrow if there's a meaningful change
    if arrow_length > 1.0:  # Higher threshold to show fewer arrows
        ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                   arrowprops=dict(arrowstyle='->', color='darkslategray', lw=0.8, alpha=0.4))

# Create legend with example point sizes
legend_sizes = [30, 100, 1000, 10000, 100000]
size_legend_elements = []
for size in legend_sizes:
    if size == 30:
        visual_size = min_size
        label = '<45 voters'
    else:
        size_capped = max(size, min_voters_threshold)
        log_size = np.log10(size_capped)
        normalized_size = (log_size - log_min) / (log_max - log_min)
        visual_size = min_size + normalized_size * (max_size - min_size)
        label = f'{size:,} voters'
    
    size_legend_elements.append(
        plt.scatter([], [], s=visual_size, c='black', alpha=0.7, 
                   edgecolors='darkgray', linewidth=0.5, 
                   label=label)
    )

# Add size legend
legend1 = ax.legend(handles=size_legend_elements, title='Mean Number of Voters', 
                   loc='lower right', framealpha=0.9, fontsize=18, title_fontsize=18,
                   columnspacing=2, handletextpad=1.5, labelspacing=1)

# Add before/after legend with arrow indication
state_legend_elements = [
    plt.Line2D([0], [0], marker='o', color='lightgray', linestyle='None', 
              markersize=8, alpha=0.5, label='Raw tally'),
    plt.Line2D([0], [0], marker='o', color='black', linestyle='None', 
              markersize=8, alpha=0.7, label='Noised tally'),
    plt.Line2D([0], [0], color='darkslategray', linestyle='-', linewidth=1, 
              label='Noise impact', alpha=0.4)
]
legend2 = ax.legend(handles=state_legend_elements, loc='center right', framealpha=0.9, fontsize=18,
                   columnspacing=2, handletextpad=1.5, labelspacing=1)
ax.add_artist(legend1)  # Re-add the first legend

# Set labels and title
ax.set_xlabel('Mean ballots leaked (%)', fontsize=25)
ax.set_ylabel('Mean voting weight leaked (%)', fontsize=25)
ax.grid(True, alpha=0.3)

# Add padding to prevent circle clipping
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05
ax.set_xlim(x_min - x_padding, x_max + x_padding)
ax.set_ylim(0 - y_padding, y_max + y_padding)

# Increase tick label font sizes
ax.tick_params(axis='both', which='major', labelsize=20)


# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

plt.tight_layout()
plt.savefig('plots/noise_impact_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("Noise Impact Summary:")
print(f"Number of DAOs compared: {len(comparison)}")

# Calculate overall impact
overall_voter_reduction = comparison['voters_deanon_percentage_normal'].mean() - comparison['voters_deanon_percentage_noise'].mean()
overall_power_reduction = comparison['voting_power_deanon_percentage_normal'].mean() - comparison['voting_power_deanon_percentage_noise'].mean()

print(f"Average reduction in correct voter deanonymization: {overall_voter_reduction:.2f} percentage points")
print(f"Average reduction in correct voting power deanonymization: {overall_power_reduction:.2f} percentage points")

# Show top 10 most affected DAOs
comparison['total_reduction'] = (comparison['voters_deanon_percentage_normal'] - comparison['voters_deanon_percentage_noise']) + (comparison['voting_power_deanon_percentage_normal'] - comparison['voting_power_deanon_percentage_noise'])
most_affected = comparison.nlargest(10, 'total_reduction')[['dao_id', 'voters_deanon_percentage_normal', 'voters_deanon_percentage_noise', 'voting_power_deanon_percentage_normal', 'voting_power_deanon_percentage_noise', 'total_reduction']]
print("\nTop 10 DAOs most affected by noise:")
print(most_affected)