import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 70)
print("Patient Distance to Attended Hospital City by Journey Type")
print("=" * 70)

# Create output directory
output_dir = 'geographic_distance_analysis'
os.makedirs(output_dir, exist_ok=True)

# Load data
print("\nLoading data...")
journey_patients = pd.read_csv('journey_umap_patients.csv')
encounters = pd.read_csv('encounters.csv')
departments = pd.read_csv('departments.csv')
patients = pd.read_csv('patients.csv')
census_codes = pd.read_csv('tigercensuscodes.csv')

print(f"Journey patients: {len(journey_patients)}")
print(f"Encounters: {len(encounters)}")
print(f"Departments: {len(departments)}")
print(f"Patients: {len(patients)}")

# Get unique departments visited by each patient
print("\nGetting departments visited by each patient...")
valid_encounters = encounters[
    (encounters['DepartmentKey'] > 0) &
    (encounters['DepartmentKey'] != -1)
].copy()

# Convert AdmissionInstant to datetime and sort by date
valid_encounters['AdmissionInstant'] = pd.to_datetime(valid_encounters['AdmissionInstant'], errors='coerce')
valid_encounters = valid_encounters.sort_values('AdmissionInstant')

# Get the LAST visited department for each patient
patient_departments = valid_encounters.groupby('PatientDurableKey').tail(1)[['PatientDurableKey', 'DepartmentKey']].drop_duplicates(subset=['PatientDurableKey'], keep='last')
patient_departments.columns = ['PatientDurableKey', 'DepartmentKey']

print(f"Unique patients with last visited department info: {len(patient_departments)}")

# Merge with departments to get city
patient_departments = patient_departments.merge(
    departments[['DepartmentKey', 'City', 'DepartmentName']],
    on='DepartmentKey',
    how='left'
)

print(f"Patients with city info: {len(patient_departments)}")

# Filter out invalid cities
patient_departments = patient_departments[
    (patient_departments['City'] != '*Unspecified') &
    (patient_departments['City'] != '*Deleted') &
    (patient_departments['City'] != '*Unknown')
].copy()

# Standardize city names to title case to fix duplicates
patient_departments['City'] = patient_departments['City'].str.title()

print(f"Patients with valid city info: {len(patient_departments)}")

# Get cities with Stormont Vail Health hospitals (including Cotton O'Neil)
stormont_cities = departments[
    (departments['DepartmentName'].str.contains('STORMONT', case=False, na=False)) |
    (departments['DepartmentName'].str.contains('COTTON', case=False, na=False))
]['City'].unique()
stormont_cities = [city for city in stormont_cities if city not in ['*Unspecified', '*Deleted', '*Unknown'] and pd.notna(city)]
stormont_cities = [city.title() for city in stormont_cities]

# Add all cities from Stormont Vail Health system (excluding Lawrence due to insufficient data)
all_stormont_cities = ['Topeka', 'Carbondale', 'Emporia', 'Meriden', 'Netawaka', 
                        'Osage City', 'Oskaloosa', 'Wamego', 'Junction City', 'Manhattan']
stormont_cities = sorted(set(stormont_cities + all_stormont_cities))

print(f"\nStormont Vail Health cities (from data and provided list):")
for i, city in enumerate(stormont_cities, 1):
    print(f"{i}. {city}")

# Filter to Stormont Vail cities
patient_departments_top = patient_departments[patient_departments['City'].isin(stormont_cities)].copy()

# Merge with journey types
patient_journey_city = patient_departments_top.merge(
    journey_patients[['PatientDurableKey', 'Journey_Type']],
    on='PatientDurableKey',
    how='inner'
)

print(f"Patients with journey type and city: {len(patient_journey_city)}")

# Get patient home coordinates
patient_journey_city = patient_journey_city.merge(
    patients[['DurableKey', 'CensusBlockGroupFipsCode']],
    left_on='PatientDurableKey',
    right_on='DurableKey',
    how='inner'
)

# Convert to string for merging
patient_journey_city['CensusBlockGroupFipsCode'] = patient_journey_city['CensusBlockGroupFipsCode'].astype(str)
census_codes['GEOID'] = census_codes['GEOID'].astype(str)

patient_journey_city = patient_journey_city.merge(
    census_codes[['GEOID', 'CENTLAT', 'CENTLON']],
    left_on='CensusBlockGroupFipsCode',
    right_on='GEOID',
    how='inner'
)

print(f"Patients with home coordinates: {len(patient_journey_city)}")

# City coordinates found online
city_coordinates = {
    'Topeka': (39.0483, -95.6780),
    'TOPEKA': (39.0483, -95.6780),
    'Carbondale': (38.5400, -95.6900),
    'CARBONDALE': (38.5400, -95.6900),
    'Emporia': (38.4059, -96.1883),
    'EMPORIA': (38.4059, -96.1883),
    'Lawrence': (38.9717, -95.2353),
    'LAWRENCE': (38.9717, -95.2353),
    'Meriden': (38.7917, -96.0750),
    'MERIDEN': (38.7917, -96.0750),
    'Netawaka': (39.5617, -95.6175),
    'NETAWAKA': (39.5617, -95.6175),
    'Osage City': (38.6325, -95.8250),
    'OSAGE CITY': (38.6325, -95.8250),
    'Oskaloosa': (39.2167, -95.3083),
    'OSKALOOSA': (39.2167, -95.3083),
    'Wamego': (39.2089, -96.3053),
    'WAMEGO': (39.2089, -96.3053),
    'Junction City': (39.0266, -96.8528),
    'JUNCTION CITY': (39.0266, -96.8528),
    'Manhattan': (39.1836, -96.5717),
    'MANHATTAN': (39.1836, -96.5717)
}

# Haversine distance function
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth radius in km
    return c * r

# Calculate distance to attended city
def calculate_distance_to_city(row):
    city = row['City']
    if city in city_coordinates:
        city_lat, city_lon = city_coordinates[city]
        return haversine_distance(row['CENTLAT'], row['CENTLON'], city_lat, city_lon)
    return np.nan

patient_journey_city['Distance_to_Attended_City_km'] = patient_journey_city.apply(calculate_distance_to_city, axis=1)

# Drop NaN distances
patient_journey_city_clean = patient_journey_city.dropna(subset=['Distance_to_Attended_City_km'])
print(f"Patients with valid distances: {len(patient_journey_city_clean)}")

# Calculate statistics by city and journey type
city_journey_stats = patient_journey_city_clean.groupby(['City', 'Journey_Type'])['Distance_to_Attended_City_km'].agg(['mean', 'median', 'std', 'count'])
city_journey_stats = city_journey_stats.reset_index()

print("\nDistance statistics by city and journey type:")
print(city_journey_stats[['City', 'Journey_Type', 'mean', 'median', 'count']])

# Create combined box and whisker plot
fig, ax = plt.subplots(figsize=(16, 8))

# Prepare data for plotting
cities_to_plot = [city for city in stormont_cities if city in city_coordinates]
journey_types = ['Escalator (Care Escalation)', 'Cycler (Clinical Redirection)', 'Abandoned (Static Journey)']

# Create positions for boxes - grouped closer together
n_cities = len(cities_to_plot)
n_journey_types = len(journey_types)
box_width = 0.8
group_spacing = 0.5
within_group_spacing = 0.0
positions = []
box_widths = []  # Store individual box widths

data_to_plot = []
labels = []
group_labels = []
group_positions = []

for i, city in enumerate(cities_to_plot):
    group_center = i * (n_journey_types * box_width + group_spacing) + 1
    
    for j, journey_type in enumerate(journey_types):
        city_data = patient_journey_city_clean[
            (patient_journey_city_clean['City'] == city) &
            (patient_journey_city_clean['Journey_Type'] == journey_type)
        ]['Distance_to_Attended_City_km'].values

        if len(city_data) > 10:
            data_to_plot.append(city_data)
            pos = group_center + (j - 1) * (box_width + within_group_spacing)
            positions.append(pos)
            labels.append(journey_type.split(' (')[0])
            box_widths.append(box_width)

    group_labels.append(city)
    group_positions.append(group_center)

# Create box plot
if data_to_plot:
    boxplot = ax.boxplot(data_to_plot, positions=positions, widths=box_width, patch_artist=True, showfliers=False,
                        medianprops={'color': 'black', 'linewidth': 2},
                        whiskerprops={'linewidth': 1.5},
                        capprops={'linewidth': 1.5})
    
    # Color the boxes: Escalator=light green, Cycler=light yellow, Abandoned=light red
    # Outline colors: darker green, darker yellow, darker red
    fill_colors = []
    edge_colors = []
    for i, city in enumerate(cities_to_plot):
        fill_colors.extend(['#6bcb77', '#ffd93d', '#ff6b6b'])
        edge_colors.extend(['#0d3d0d', '#806600', '#8b0000'])
    
    for patch, fill_color, edge_color in zip(boxplot['boxes'], fill_colors, edge_colors):
        patch.set_facecolor(fill_color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(edge_color)
        patch.set_linewidth(1.5)
    
    # Set whisker and cap colors to match edge colors
    for i, edge_color in enumerate(edge_colors):
        boxplot['whiskers'][i*2].set_color(edge_color)
        boxplot['whiskers'][i*2+1].set_color(edge_color)
        boxplot['caps'][i*2].set_color(edge_color)
        boxplot['caps'][i*2+1].set_color(edge_color)
    
    # Set x-axis labels - show city names as group labels horizontally
    ax.set_xticks(group_positions)
    ax.set_xticklabels(group_labels, rotation=0, ha='center', fontsize=11)
    
    # Add legend for journey types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#6bcb77', edgecolor='black', label='Escalator'),
        Patch(facecolor='#ffd93d', edgecolor='black', label='Cycler'),
        Patch(facecolor='#ff6b6b', edgecolor='black', label='Static')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    
    ax.set_ylabel('Distance to Hospital City (km)', fontsize=14)
    ax.set_xlabel('City', fontsize=14)
    ax.set_title('Diabetes Patient Distance to Hospital by Journey Type', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = f'{output_dir}/distance_to_attended_city_boxplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved boxplot to: {output_path}")
    
    city_journey_stats.to_csv(f'{output_dir}/distance_to_attended_city_stats.csv', index=False)
    print(f"Saved statistics to: {output_dir}/distance_to_attended_city_stats.csv")
else:
    print("No data to plot")

print("\n" + "=" * 70)
print("Analysis Complete")
print("=" * 70)
print("\nColor coding:")
print("Red: Abandoned (Static Journey)")
print("Yellow: Cycler (Clinical Redirection)")
print("Green: Escalator (Care Escalation)")
