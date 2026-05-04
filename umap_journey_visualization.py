import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

print("=" * 70)
print("UMAP Journey Type Visualization with Loose Light Shading")
print("=" * 70)

# Load existing UMAP data
print("\nLoading existing UMAP data...")
journey_data = pd.read_csv('journey_umap_patients.csv')
print(f"Loaded {len(journey_data)} patients with UMAP coordinates")

# Check if UMAP coordinates exist
if 'UMAP_1' not in journey_data.columns or 'UMAP_2' not in journey_data.columns:
    print("Error: UMAP coordinates not found in journey_umap_patients.csv")
    print("Please run the full UMAP analysis first")
    exit(1)

# Create output folder
import os
output_base = 'output_journey_umap'
os.makedirs(output_base, exist_ok=True)

print(f"\nOutput folder: {output_base}/")

# Define colors for journey types
journey_colors = {
    'Escalator (Care Escalation)': '#2ecc71',
    'Cycler (Clinical Redirection)': '#f39c12',
    'Static (Static Journey)': '#e74c3c'
}

# Create figure
plt.figure(figsize=(14, 10))

# First, draw KDE shading around each cluster with loose light shading
for journey_type, color in journey_colors.items():
    mask = journey_data['Journey_Type'] == journey_type
    x = journey_data.loc[mask, 'UMAP_1'].values
    y = journey_data.loc[mask, 'UMAP_2'].values
    
    print(f"\n{journey_type}: {len(x)} points")
    
    if len(x) > 10:  # Need enough points for KDE
        try:
            # Create grid for contour plot
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
            
            # Extend grid differently for each journey type
            x_range = xmax - xmin
            y_range = ymax - ymin
            if journey_type == 'Static (Static Journey)':
                xmin -= x_range * 0.15
                xmax += x_range * 0.15
                ymin -= y_range * 0.15
                ymax += y_range * 0.15
            elif journey_type == 'Cycler (Clinical Redirection)':
                xmin -= x_range * 0.08
                xmax += x_range * 0.08
                ymin -= y_range * 0.08
                ymax += y_range * 0.08
            else:
                xmin -= x_range * 0.1
                xmax += x_range * 0.1
                ymin -= y_range * 0.1
                ymax += y_range * 0.1
            
            # Create grid
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            
            # Calculate KDE
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([x, y])
            
            # Use different bandwidths for each journey type
            if journey_type == 'Static (Static Journey)':
                bandwidth = 0.32  # Slightly smaller to reduce red area
            elif journey_type == 'Cycler (Clinical Redirection)':
                bandwidth = 0.30  # Even larger bandwidth for more uniform yellow shading
            else:
                bandwidth = 0.30  # Larger bandwidth for green area
            
            kernel = gaussian_kde(values, bw_method=bandwidth)
            zz = np.reshape(kernel(positions).T, xx.shape)
            
            # Calculate contours for shading (don't plot the lines)
            levels = 12  # Even more levels to better capture cluster shape and curvature
            contours = plt.contour(xx, yy, zz, levels=levels, colors=color, alpha=0)
            
            # Add shading with tighter parameters to follow cluster shape
            if journey_type == 'Static (Static Journey)':
                start_level = 1  # Start from outer levels to capture full shape
                base_alpha = 0.12  # Reduced contrast
            elif journey_type == 'Cycler (Clinical Redirection)':
                start_level = 1  # Start from outer levels to capture full shape
                base_alpha = 0.20  # Base opacity
            else:
                start_level = 1  # Start from outer levels to capture full shape
                base_alpha = 0.12  # Reduced contrast
            
            for level_idx, path_collection in enumerate(contours.allsegs):
                if level_idx >= start_level:
                    # Decrease opacity of second last layer for yellow
                    if journey_type == 'Cycler (Clinical Redirection)' and level_idx == len(contours.allsegs) - 2:
                        alpha = base_alpha * 0.02  # Make second last layer nearly invisible
                    else:
                        alpha = base_alpha
                    for path in path_collection:
                        if len(path) > 2:
                            poly = plt.Polygon(path, facecolor=color, alpha=alpha, edgecolor='none')
                            plt.gca().add_patch(poly)
        except Exception as e:
            print(f"  Warning: Could not create shading for {journey_type}: {e}")
            pass

# Then plot all scatter points
for journey_type, color in journey_colors.items():
    mask = journey_data['Journey_Type'] == journey_type
    plt.scatter(
        journey_data.loc[mask, 'UMAP_1'],
        journey_data.loc[mask, 'UMAP_2'],
        c=color,
        label=journey_type,
        s=30,
        alpha=0.5,
        edgecolor='none'
    )

plt.title('Diabetes Patient Journey Type UMAP', fontsize=16, fontweight='bold')
plt.xlabel('UMAP_1', fontsize=14)
plt.ylabel('UMAP_2', fontsize=14)
plt.legend(title='Journey Type', title_fontsize=12, fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout(rect=[0, 0, 0.85, 1])

plot_path = f'{output_base}/umap_journey_types_loose_shading.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nSaved UMAP plot with loose shading to {plot_path}")
plt.close()

print("\n" + "=" * 70)
print("Visualization Complete!")
print("=" * 70)
