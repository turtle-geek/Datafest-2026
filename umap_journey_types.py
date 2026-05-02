import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
from sklearn.manifold import trustworthiness
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

print("=" * 70)
print("UMAP - Journey Type Clustering")
print("Visualizing Patient Clustering by Journey Type")
print("=" * 70)

# Create output folder
output_base = 'output_journey_umap'
os.makedirs(output_base, exist_ok=True)

print(f"\nCreated output folder: {output_base}/")

# Load data
print("\nLoading data files...")
encounters = pd.read_csv('encounters.csv')
patients = pd.read_csv('patients.csv')
diagnosis = pd.read_csv('diagnosis.csv')
departments = pd.read_csv('departments.csv')
providers = pd.read_csv('providers.csv')

print(f"Encounters: {len(encounters)} rows")
print(f"Patients: {len(patients)} rows")
print(f"Diagnosis: {len(diagnosis)} rows")
print(f"Departments: {len(departments)} rows")
print(f"Providers: {len(providers)} rows")

# Join encounters with diagnosis
print("\nJoining data...")
encounters_diag = encounters.merge(
    diagnosis[['DiagnosisKey', 'DiagnosisValue']],
    left_on='PrimaryDiagnosisKey',
    right_on='DiagnosisKey',
    how='left'
)

# Join with patients
encounters_diag_pat = encounters_diag.merge(
    patients[['DurableKey']],
    left_on='PatientDurableKey',
    right_on='DurableKey',
    how='left'
)

# Join with departments
encounters_diag_pat_dept = encounters_diag_pat.merge(
    departments[['DepartmentKey', 'DepartmentSpecialty', 'DepartmentType']],
    left_on='DepartmentKey',
    right_on='DepartmentKey',
    how='left'
)

# Join with providers
encounters_full = encounters_diag_pat_dept.merge(
    providers[['DurableKey', 'PrimarySpecialty', 'Type']].rename(columns={
        'DurableKey': 'ProviderDurableKey_Ref',
        'PrimarySpecialty': 'ProviderSpecialty',
        'Type': 'ProviderType'
    }),
    left_on='ProviderDurableKey',
    right_on='ProviderDurableKey_Ref',
    how='left'
)

# Filter for Diabetes (all types: E10, E11, E13, E14)
print("\nFiltering for Diabetes patients (all types: E10, E11, E13, E14)...")
diabetes_mask = encounters_full['DiagnosisValue'].str.startswith(('E10', 'E11', 'E13', 'E14'), na=False)
diabetes_data = encounters_full[diabetes_mask].copy()

print(f"Diabetes encounters: {len(diabetes_data)} rows")
print(f"Unique diabetes patients: {diabetes_data['PatientDurableKey'].nunique()}")

# Convert Date to datetime
diabetes_data['Date'] = pd.to_datetime(diabetes_data['Date'], errors='coerce')
diabetes_data = diabetes_data.sort_values(['PatientDurableKey', 'Date'])

# Define Journey Types (same as treatment path mapping)
print("\n" + "=" * 70)
print("Defining Journey Types Based on Treatment Paths")
print("=" * 70)

def classify_journey_type(group):
    group = group.sort_values('Date')
    group = group[group['Date'].notna()]
    
    if len(group) < 3:
        return 'Abandoned (Static Journey)'
    
    dept_counts = group['DepartmentSpecialty'].value_counts()
    unique_depts = len(dept_counts)
    redundancy = (dept_counts > 1).sum()
    
    if redundancy >= 2:
        return 'Cycler (Clinical Redirection)'
    
    if unique_depts >= 3:
        return 'Escalator (Care Escalation)'
    
    if unique_depts >= 2:
        return 'Escalator (Care Escalation)'
    else:
        return 'Cycler (Clinical Redirection)'

print("Classifying patient journeys...")
journey_types = diabetes_data.groupby('PatientDurableKey').apply(classify_journey_type).reset_index()
journey_types.columns = ['PatientDurableKey', 'Journey_Type']

print(f"\nJourney Type Distribution:")
print(journey_types['Journey_Type'].value_counts())

# Calculate features per patient (same as enhanced UMAP)
print("\n" + "=" * 70)
print("Calculating Features per Patient")
print("=" * 70)

def calculate_features(group):
    group = group.sort_values('Date')
    dates = group['Date'].dropna()
    
    if len(dates) < 1:
        return pd.Series({
            'Velocity': np.nan,
            'Redundancy': 0,
            'Latency': 0,
            'Total_Encounters': len(group),
            'Dept_Diversity': 0,
            'Provider_Diversity': 0,
            'ED_Ratio': 0,
            'ICU_Ratio': 0,
            'Specialty_Diversity': 0
        })
    
    if len(dates) >= 2:
        date_diffs = dates.diff().dt.days.dropna()
        velocity = date_diffs.mean() if len(date_diffs) > 0 else np.nan
    else:
        velocity = 0
    
    dept_counts = group['DepartmentKey'].value_counts()
    redundancy = (dept_counts > 1).sum()
    
    if len(dates) >= 2:
        latency = (dates.max() - dates.min()).days
    else:
        latency = 0
    
    dept_specialties = group['DepartmentSpecialty'].dropna().unique()
    dept_diversity = len(dept_specialties) if len(dept_specialties) > 0 else 0
    
    provider_specialties = group['ProviderSpecialty'].dropna().unique()
    provider_diversity = len(provider_specialties) if len(provider_specialties) > 0 else 0
    
    ed_visits = (group['DepartmentType'] == 'ED').sum()
    ed_ratio = ed_visits / len(group) if len(group) > 0 else 0
    
    icu_visits = (group['DepartmentType'] == 'ICU').sum()
    icu_ratio = icu_visits / len(group) if len(group) > 0 else 0
    
    provider_types = group['ProviderType'].dropna().unique()
    specialty_diversity = len(provider_types) if len(provider_types) > 0 else 0
    
    return pd.Series({
        'Velocity': velocity,
        'Redundancy': redundancy,
        'Latency': latency,
        'Total_Encounters': len(group),
        'Dept_Diversity': dept_diversity,
        'Provider_Diversity': provider_diversity,
        'ED_Ratio': ed_ratio,
        'ICU_Ratio': icu_ratio,
        'Specialty_Diversity': specialty_diversity
    })

print("Calculating features...")
patient_features = diabetes_data.groupby('PatientDurableKey').apply(calculate_features).reset_index()

# Fill missing Velocity with median
velocity_median = patient_features['Velocity'].median()
patient_features['Velocity'] = patient_features['Velocity'].fillna(velocity_median)

print(f"Patients with features: {len(patient_features)}")

# Merge with journey types
patient_features = patient_features.merge(journey_types, on='PatientDurableKey', how='left')

# Select features for UMAP
feature_cols = ['Velocity', 'Redundancy', 'Latency', 'Dept_Diversity', 'Provider_Diversity', 
                'ED_Ratio', 'ICU_Ratio', 'Specialty_Diversity']

X = patient_features[feature_cols].values

# Encode journey types as numeric labels
journey_type_mapping = {
    'Abandoned (Static Journey)': 0,
    'Cycler (Clinical Redirection)': 1,
    'Escalator (Care Escalation)': 2
}
y = patient_features['Journey_Type'].map(journey_type_mapping).values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run Supervised UMAP with Journey Type labels
print("\n" + "=" * 70)
print("Running Supervised UMAP with Journey Type Labels")
print("=" * 70)

reducer = umap.UMAP(
    n_neighbors=50,
    min_dist=0.01,
    n_epochs=500,
    target_metric='categorical',
    random_state=42,
    n_jobs=1
)

X_umap = reducer.fit_transform(X_scaled, y=y)
print(f"UMAP projection shape: {X_umap.shape}")

# Calculate Embedding Trustworthiness
trust_score = trustworthiness(X_scaled, X_umap, n_neighbors=50)
print(f"Embedding Trustworthiness: {trust_score:.4f}")

# Add UMAP coordinates to dataframe
patient_features['UMAP_1'] = X_umap[:, 0]
patient_features['UMAP_2'] = X_umap[:, 1]

# Visualize UMAP colored by Journey Type with Refined Cluster Shading
print("\n" + "=" * 70)
print("Visualizing UMAP by Journey Type with Refined Cluster Shading")
print("=" * 70)

from scipy.stats import gaussian_kde

plt.figure(figsize=(14, 10))

# Define colors for journey types
journey_colors = {
    'Abandoned (Static Journey)': '#e74c3c',
    'Cycler (Clinical Redirection)': '#f39c12',
    'Escalator (Care Escalation)': '#2ecc71'
}

# First, draw KDE shading around each cluster using refined parameters
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
            # For red clusters, shade inner levels for compact shading but ensure at least 2 circles
            # For yellow and green, shade more levels for increased shading depth
            if journey_type == 'Abandoned (Static Journey)':
                start_level = 3  # Inner levels for compact red shading with at least 2 circles
                alpha = 0.12
            elif journey_type == 'Cycler (Clinical Redirection)':
                start_level = 2  # Reduce yellow shading to be more compact
                alpha = 0.20  # Higher contrast for yellow
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
    plt.scatter(
        patient_features.loc[mask, 'UMAP_1'],
        patient_features.loc[mask, 'UMAP_2'],
        c=color,
        label=journey_type,
        s=40,
        alpha=0.4,
        edgecolor='none'
    )

plt.title('Diabetes Patient Journey Type UMAP', fontsize=16, fontweight='bold')
plt.xlabel('UMAP_1', fontsize=14)
plt.ylabel('UMAP_2', fontsize=14)
plt.legend(title='Journey Type', title_fontsize=12, fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend on the right

plot_path = f'{output_base}/umap_journey_types.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved UMAP plot with refined shading to {plot_path}")
plt.close()

# Create feature importance analysis by journey type
print("\n" + "=" * 70)
print("Feature Analysis by Journey Type")
print("=" * 70)

feature_means = patient_features.groupby('Journey_Type')[feature_cols].mean()
print("\nMean feature values by journey type:")
print(feature_means)

# Visualize feature differences
plt.figure(figsize=(14, 8))
feature_means.T.plot(kind='bar', figsize=(14, 8), colormap='viridis')
plt.title('Feature Means by Journey Type', fontsize=16, fontweight='bold')
plt.ylabel('Mean Value', fontsize=14)
plt.xlabel('Feature', fontsize=14)
plt.legend(title='Journey Type', title_fontsize=12, fontsize=11, loc='best')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

feature_means_path = f'{output_base}/feature_means_by_journey_type.png'
plt.savefig(feature_means_path, dpi=300, bbox_inches='tight')
print(f"Saved feature means plot to {feature_means_path}")
plt.close()

# Save outputs
print("\n" + "=" * 70)
print("Saving Outputs")
print("=" * 70)

# Save model
with open('team21_journey_umap.pkl', 'wb') as f:
    pickle.dump(reducer, f)
print("Saved UMAP model to team21_journey_umap.pkl")

# Save scaler
with open('team21_journey_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved scaler to team21_journey_scaler.pkl")

# Save patient data
output_cols = ['PatientDurableKey', 'Journey_Type'] + feature_cols + ['UMAP_1', 'UMAP_2']
patient_features[output_cols].to_csv('journey_umap_patients.csv', index=False)
print(f"Saved {len(patient_features)} patient records to journey_umap_patients.csv")

# Save feature means
feature_means.to_csv(f'{output_base}/feature_means_by_journey_type.csv')
print(f"Saved feature means to {output_base}/feature_means_by_journey_type.csv")

print("\n" + "=" * 70)
print("UMAP Journey Type Clustering Complete!")
print("=" * 70)
print(f"\nFiles generated:")
print(f"1. team21_journey_umap.pkl - Trained UMAP model")
print(f"2. team21_journey_scaler.pkl - Feature scaler")
print(f"3. journey_umap_patients.csv - Patient data with journey types")
print(f"4. {output_base}/umap_journey_types.png - UMAP visualization by journey type")
print(f"5. {output_base}/feature_means_by_journey_type.png - Feature means plot")
print(f"6. {output_base}/feature_means_by_journey_type.csv - Feature means data")
print(f"\nEmbedding Trustworthiness: {trust_score:.4f}")
print(f"\nJourney Type Distribution:")
for journey_type, count in patient_features['Journey_Type'].value_counts().items():
    print(f"  {journey_type}: {count} patients ({count/len(patient_features)*100:.1f}%)")
