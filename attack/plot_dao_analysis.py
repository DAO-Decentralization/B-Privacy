import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV file
df = pd.read_csv('privacy_attack_results.csv')

# Calculate deanonymization percentage and option1 percentage
df['deanon_percentage'] = (df['total_voting_power_deanonymized'] / df['total_voting_power']) * 100
df['option1_margin'] = ((df['option1_votes'] - df['option2_votes'])/ df['total_voting_power']) * 100
df['option1_margin'] = df['option1_margin'].fillna(100)
# Specify the 4 DAOs to analyze
target_daos = ['arbitrumfoundation.eth', 'aavegotchi.eth', 'sushigov.eth', 'balancer.eth']
dao_avg_voters = df.groupby('dao_id')['total_voters'].mean()
top_4_daos = target_daos

# Create subplots - single row for paper layout with extra width for colorbar
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

# Store scatter plot for shared colorbar
main_scatter = None

for i, dao_id in enumerate(top_4_daos):
    dao_data = df[df['dao_id'] == dao_id]
    dao_name = {
            'arbitrumfoundation.eth': 'Arbitrum', 
            'sushigov.eth': 'Sushi',
            'aavegotchi.eth': 'Aavegotchi', 
            'balancer.eth': 'Balancer'
        }.get(dao_id)
    
    # Plot points with color based on option1_percentage
    scatter = axes[i].scatter(
        dao_data['total_voters'], 
        dao_data['deanon_percentage'],
        c=dao_data['option1_margin'],
        s=80,  # Fixed size
        alpha=0.7,
        cmap='viridis',
        vmin=50,
        vmax=100,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Store scatter for shared colorbar
    if i == 0:  # Only need one reference for colorbar
        main_scatter = scatter
    
    axes[i].set_xlabel('Proposal voter count', fontsize=15)
    if i == 0:  # Only label y-axis on leftmost plot
        axes[i].set_ylabel('Voting weight leaked (%)', fontsize=15)
    axes[i].set_title(f'{dao_name}\n(Mean voters: {dao_avg_voters[dao_id]:.0f})', fontsize=15)
    axes[i].grid(True, alpha=0.3)

    # Increase tick label font sizes
    axes[i].tick_params(axis='both', which='major', labelsize=15)
    
    # Set consistent y-axis scale for all subplots
    axes[i].set_ylim(0, 100)
    
    # Remove y-tick labels from all but the first subplot
    if i > 0:
        axes[i].tick_params(axis='y', labelleft=False)
    
    # Improve x-axis formatting, especially for arbitrum
    axes[i].tick_params(axis='x', rotation=45)

# Add padding to each subplot to prevent circle clipping
for i in range(len(top_4_daos)):
    x_min, x_max = axes[i].get_xlim()
    y_min, y_max = axes[i].get_ylim()
    x_padding = (x_max - x_min) * 0.05  # 5% padding
    y_padding = (y_max - y_min) * 0.05  # 5% padding
    axes[i].set_xlim(x_min - x_padding, x_max + x_padding)
    axes[i].set_ylim(y_min - y_padding, y_max + y_padding)

# Adjust subplot spacing to make room for colorbar
plt.subplots_adjust(right=0.85)

# Add single colorbar for all subplots - positioned to the right with more space
cbar = fig.colorbar(main_scatter, ax=axes, shrink=0.8, pad=0.01, aspect=20)
cbar.set_label('Winning Margin (%)', rotation=270, labelpad=20, fontsize=15)
cbar.ax.tick_params(labelsize=15)

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/dao_deanonymization_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Selected DAOs:")
for dao_id in top_4_daos:
    print(f"{dao_id}: {dao_avg_voters[dao_id]:.0f} avg voters")