import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('results/hermes_massclass_fits.csv')

# Get unique N values
n_values = sorted(df['N'].unique())

# Create plots for each N value
for n in n_values:
    # Filter data for current N
    df_n = df[df['N'] == n]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'N = {n}', fontsize=16, fontweight='bold')
    
    # Define colors for each class
    class_colors = {'S1': 'blue', 'S2': 'red', 'S3': 'green', 'S4': 'orange'}
    
    # Plot 1: alpha_sd vs L_logM
    for class_label in df_n['class_label'].unique():
        class_data = df_n[df_n['class_label'] == class_label]
        axes[0].scatter(class_data['L_logM'], class_data['alpha_sd'], 
                       c=class_colors[class_label], label=class_label, alpha=0.7, s=60)
    
    axes[0].set_xlabel('L_logM')
    axes[0].set_ylabel('alpha_sd')
    axes[0].set_title(f'alpha_sd vs L_logM (N={n})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: beta_sd vs L_logM
    for class_label in df_n['class_label'].unique():
        class_data = df_n[df_n['class_label'] == class_label]
        axes[1].scatter(class_data['L_logM'], class_data['beta_sd'], 
                       c=class_colors[class_label], label=class_label, alpha=0.7, s=60)
    
    axes[1].set_xlabel('L_logM')
    axes[1].set_ylabel('beta_sd')
    axes[1].set_title(f'beta_sd vs L_logM (N={n})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: sigma_sd vs L_logM
    for class_label in df_n['class_label'].unique():
        class_data = df_n[df_n['class_label'] == class_label]
        axes[2].scatter(class_data['L_logM'], class_data['sigma_sd'], 
                       c=class_colors[class_label], label=class_label, alpha=0.7, s=60)
    
    axes[2].set_xlabel('L_logM')
    axes[2].set_ylabel('sigma_sd')
    axes[2].set_title(f'sigma_sd vs L_logM (N={n})')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Alternative: All plots in one figure with subplots
fig, axes = plt.subplots(len(n_values), 3, figsize=(18, 5*len(n_values)))
fig.suptitle('Standard Deviation Parameters vs L_logM by Survey Class', fontsize=16, fontweight='bold')

if len(n_values) == 1:
    axes = [axes]  # Make it 2D for consistent indexing

for i, n in enumerate(n_values):
    df_n = df[df['N'] == n]
    
    # alpha_sd vs L_logM
    for class_label in df_n['class_label'].unique():
        class_data = df_n[df_n['class_label'] == class_label]
        axes[i, 0].scatter(class_data['L_logM'], class_data['alpha_sd'], 
                          c=class_colors[class_label], label=class_label, alpha=0.7, s=60)
    axes[i, 0].set_ylabel(f'N={n}\nalpha_sd')
    axes[i, 0].grid(True, alpha=0.3)
    if i == 0:
        axes[i, 0].set_title('alpha_sd vs L_logM')
        axes[i, 0].legend()
    
    # beta_sd vs L_logM
    for class_label in df_n['class_label'].unique():
        class_data = df_n[df_n['class_label'] == class_label]
        axes[i, 1].scatter(class_data['L_logM'], class_data['beta_sd'], 
                          c=class_colors[class_label], label=class_label, alpha=0.7, s=60)
    axes[i, 1].set_ylabel(f'N={n}\nbeta_sd')
    axes[i, 1].grid(True, alpha=0.3)
    if i == 0:
        axes[i, 1].set_title('beta_sd vs L_logM')
        axes[i, 1].legend()
    
    # sigma_sd vs L_logM
    for class_label in df_n['class_label'].unique():
        class_data = df_n[df_n['class_label'] == class_label]
        axes[i, 2].scatter(class_data['L_logM'], class_data['sigma_sd'], 
                          c=class_colors[class_label], label=class_label, alpha=0.7, s=60)
    axes[i, 2].set_ylabel(f'N={n}\nsigma_sd')
    axes[i, 2].grid(True, alpha=0.3)
    if i == 0:
        axes[i, 2].set_title('sigma_sd vs L_logM')
        axes[i, 2].legend()

# Set common x-labels for bottom row
for j in range(3):
    axes[-1, j].set_xlabel('L_logM')

plt.tight_layout()
plt.show()