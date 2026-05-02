import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.path import Path
import os

print("=" * 70)
print("UMAP Visualization with Contour Plots")
print("Loading existing UMAP data...")
print("=" * 70)

# Load existing UMAP data
patient_features = pd.read_csv('journey_umap_patients.csv')
print(f"Loaded {len(patient_features)} patient records")

# Create output folder
output_base = 'output_journey_umap'
os.makedirs(output_base, exist_ok=True)

# Visualize UMAP colored by Journey Type with Transparent Shading
print("\nCreating UMAP visualization with transparent cluster boundaries...")

plt.figure(figsize=(14, 10))

# Define colors for journey types
journey_colors = {
    'Abandoned (Static Journey)': '#e74c3c',
    'Cycler (Clinical Redirection)': '#f39c12',
    'Escalator (Care Escalation)': '#2ecc71'
}

# First, draw KDE contour lines (no shading) around each cluster
for journey_type, color in journey_colors.items():
    mask = patient_features['Journey_Type'] == journey_type
    x = patient_features.loc[mask, 'UMAP_1'].values
    y = patient_features.loc[mask, 'UMAP_2'].values
    
    if len(x) > 10:  # Need enough points for KDE
        try:
            # Create grid for contour plot
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
            
            # Extend grid slightly
            x_range = xmax - xmin
            y_range = ymax - ymin
            xmin -= x_range * 0.1
            xmax += x_range * 0.1
            ymin -= y_range * 0.1
            ymax += y_range * 0.1
            
            # Create grid
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            
            # Calculate KDE
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([x, y])
            
            # Use smaller bandwidth for red clusters to follow shape better
            if journey_type == 'Abandoned (Static Journey)':
                bandwidth = 0.3  # Smaller bandwidth for more detailed shape-following
            else:
                bandwidth = None  # Default bandwidth for yellow and green
            
            kernel = gaussian_kde(values, bw_method=bandwidth)
            zz = np.reshape(kernel(positions).T, xx.shape)
            
            # Calculate contours for shading (don't plot the lines)
            # Use more levels for all clusters now
            levels = 8 if journey_type == 'Abandoned (Static Journey)' else 7
            contours = plt.contour(xx, yy, zz, levels=levels, colors=color, alpha=0)  # alpha=0 to hide lines
            
            # Add shading only within the contour paths using the contour vertices
            # For red clusters, shade innermost levels for smaller shading
            # For yellow and green, shade more levels for increased shading depth
            if journey_type == 'Abandoned (Static Journey)':
                start_level = 4  # Innermost levels for smaller red shading
                alpha = 0.12
            elif journey_type == 'Cycler (Clinical Redirection)':
                start_level = 1  # Keep same number of levels
                alpha = 0.13  # Intermediate alpha for balanced yellow shading
            else:
                start_level = 1  # Shade more levels for larger green shading
                alpha = 0.12
            for level_idx, path_collection in enumerate(contours.allsegs):
                if level_idx >= start_level:
                    for path in path_collection:
                        if len(path) > 2:  # Need at least 3 points for a polygon
                            # Convert path to polygon and fill with light shading
                            poly = plt.Polygon(path, facecolor=color, alpha=alpha, edgecolor='none')
                            plt.gca().add_patch(poly)
        except:
            pass  # Skip if KDE fails

# Then plot all scatter points
for journey_type, color in journey_colors.items():
    mask = patient_features['Journey_Type'] == journey_type
    # Simplify labels by removing bracketed text
    simple_label = journey_type.split(' (')[0]
    plt.scatter(
        patient_features.loc[mask, 'UMAP_1'],
        patient_features.loc[mask, 'UMAP_2'],
        c=color,
        label=simple_label,
        s=40,
        alpha=0.4,
        edgecolor='none'
    )

plt.title('Diabetes Patient Journey Type UMAP', fontsize=16, fontweight='bold')
plt.xlabel('UMAP_1', fontsize=14)
plt.ylabel('UMAP_2', fontsize=14)
plt.legend(title='Journey Type', title_fontsize=14, fontsize=13, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, labelspacing=0.5)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout(rect=[0, 0, 0.82, 1])  # Make room for legend on the right

plot_path = f'{output_base}/umap_journey_types.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved UMAP plot with contours to {plot_path}")
plt.close()

print("\nVisualization complete!")
