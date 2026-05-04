import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# ==========================================
# CONFIG
# ==========================================
R = 10             # coverage radius in km (10km realistic for rural)
K_MAX = 80         # test up to 80 facilities if needed
TARGET_COVERAGE = 90.0  # Stop when we hit this coverage %
R_VALUES = [3, 5, 10, 15, 20]  # for comparison at the end

# ==========================================
# STEP 1 - LOAD AND CLEAN
# ==========================================
print("Loading data...")
patients = pd.read_csv("patients.csv", usecols=["DurableKey", "CensusBlockGroupFipsCode"])
departments = pd.read_csv("departments.csv")
tiger = pd.read_csv("tigercensuscodes.csv")

# --- Patient locations via census block group ---
# Filter out invalid FIPS codes
patients = patients[
    ~patients["CensusBlockGroupFipsCode"].isin(["*Unspecified", "*Not Applicable", "*Deleted"])
].copy()
patients.rename(columns={"DurableKey": "PatientDurableKey"}, inplace=True)

# Tiger GEOID is 12-digit block group; patient FIPS is also 12-digit block group
# Join directly to get patient lat/lon
tiger["GEOID_str"] = tiger["GEOID"].astype(str)
patients = patients.merge(
    tiger[["GEOID_str", "CENTLAT", "CENTLON", "PopulationValue"]],
    left_on="CensusBlockGroupFipsCode",
    right_on="GEOID_str",
    how="inner"
)
patients.rename(columns={"CENTLAT": "pat_lat", "CENTLON": "pat_lon"}, inplace=True)
patients.drop(columns=["GEOID_str"], inplace=True)
patients = patients.dropna(subset=["pat_lat", "pat_lon"])

print(f"  Patients with coordinates: {len(patients):,}")

# --- Provider (department) locations via census tract ---
# Department CensusTract is 11-digit tract; tiger GEOID is 12-digit block group
# Truncate tiger GEOID to 11 digits to match at the tract level
dept_valid = departments[departments["CensusTract"].notna()].copy()
dept_valid["tract_str"] = dept_valid["CensusTract"].astype(int).astype(str)

tiger_tracts = tiger.copy()
tiger_tracts["tract_str"] = tiger_tracts["GEOID"].astype(str).str[:11]
# Average lat/lon across block groups within same tract
tract_coords = tiger_tracts.groupby("tract_str").agg(
    prov_lat=("CENTLAT", "mean"),
    prov_lon=("CENTLON", "mean")
).reset_index()

dept_geo = dept_valid.merge(tract_coords, on="tract_str", how="inner")
# One row per department (average coords if multiple block groups matched)
providers = (
    dept_geo
    .groupby("DepartmentKey")
    .agg(
        prov_lat=("prov_lat", "mean"),
        prov_lon=("prov_lon", "mean"),
        DepartmentName=("DepartmentName", "first"),
        City=("City", "first"),
    )
    .reset_index()
    .dropna(subset=["prov_lat", "prov_lon"])
)

# Remove duplicates at same location (round to 4 decimal places ~11m precision)
providers["loc_key"] = (
    providers["prov_lat"].round(4).astype(str) + "_" +
    providers["prov_lon"].round(4).astype(str)
)
providers = providers.drop_duplicates(subset="loc_key").drop(columns="loc_key")

print(f"  Provider locations: {len(providers):,}")

# ==========================================
# STEP 2 - HAVERSINE DISTANCE (VECTORIZED)
# ==========================================
def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance in km.
    All inputs can be numpy arrays.
    """
    R_EARTH = 6371.0  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


# ==========================================
# STEP 3 & 4 - COMPUTE CURRENT ACCESS
# ==========================================
print(f"\nComputing current access (R = {R} km)...")

# Vectorized: for each patient, find distance to nearest provider
# Use chunked computation to avoid memory explosion
# (patients x providers matrix could be huge)
prov_lats = providers["prov_lat"].values
prov_lons = providers["prov_lon"].values
pat_lats = patients["pat_lat"].values
pat_lons = patients["pat_lon"].values

CHUNK = 5000
min_distances = np.empty(len(patients))

for start in range(0, len(patients), CHUNK):
    end = min(start + CHUNK, len(patients))
    # Shape: (chunk_size, num_providers)
    dists = haversine_km(
        pat_lats[start:end, np.newaxis], pat_lons[start:end, np.newaxis],
        prov_lats[np.newaxis, :],        prov_lons[np.newaxis, :]
    )
    min_distances[start:end] = dists.min(axis=1)

patients["distance_to_nearest_provider"] = min_distances
patients["is_covered"] = patients["distance_to_nearest_provider"] <= R

coverage_before = patients["is_covered"].mean() * 100
avg_dist_before = patients["distance_to_nearest_provider"].mean()

print(f"  Coverage before: {coverage_before:.1f}%")
print(f"  Avg distance to nearest provider: {avg_dist_before:.2f} km")

# ==========================================
# STEP 5 - IDENTIFY UNDERSERVED
# ==========================================
underserved = patients[patients["is_covered"] == False].copy()
print(f"  Underserved patients: {len(underserved):,} ({100 - coverage_before:.1f}%)")

# ==========================================
# STEP 6 - WEIGHT BY POPULATION
# ==========================================
print("\nWeighting by census block population...")

# Group underserved patients by their census block location
block_weights = (
    underserved
    .groupby(["pat_lat", "pat_lon"])
    .agg(
        weight=("PatientDurableKey", "count"),
        pop=("PopulationValue", "first")
    )
    .reset_index()
)

print(f"  Unique underserved locations: {len(block_weights):,}")
# ==========================================
# STEP 7 - GREEDY MAXIMUM COVERAGE ALGORITHM
# ==========================================
# This combines BOTH approaches:
# 1. It targets raw patient density (biggest groups)
# 2. It avoids overlap by only counting NEWLY covered patients
# ==========================================
print(f"\nOptimising K using Greedy Max Coverage (targeting {TARGET_COVERAGE}%)...")

# OPTIMIZATION: Group 334,000 patients into their unique 1,321 blocks for lightning-fast distance math
patient_blocks = patients.groupby(["pat_lat", "pat_lon"]).agg(
    count=("PatientDurableKey", "count"),
    min_dist=("distance_to_nearest_provider", "min")
).reset_index()

pb_lats = patient_blocks["pat_lat"].values
pb_lons = patient_blocks["pat_lon"].values
pb_counts = patient_blocks["count"].values
current_min_dist = patient_blocks["min_dist"].values.copy()

# The candidates are the coordinates of the currently underserved blocks
candidates = block_weights[["pat_lat", "pat_lon"]].values
new_facilities_list = []
results = []

print(f"  {'K':>3}  {'Coverage':>9}  {'vs Base':>9}  {'Avg Dist':>9}")
print(f"  {'-'*3}  {'-'*9}  {'-'*9}  {'-'*9}")

for k in range(1, K_MAX + 1):
    best_candidate_idx = -1
    best_additional_coverage = -1
    best_candidate_dists = None
    
    # Evaluate every candidate location against the 1,321 blocks
    for i, candidate in enumerate(candidates):
        dists = haversine_km(pb_lats, pb_lons, candidate[0], candidate[1])
        
        # Multiply boolean mask (1 or 0) by patient counts to get number of newly covered patients
        newly_covered = (((current_min_dist > R) & (dists <= R)) * pb_counts).sum()
        
        if newly_covered > best_additional_coverage:
            best_additional_coverage = newly_covered
            best_candidate_idx = i
            best_candidate_dists = dists
            
    # Place facility at the winning location
    best_cand = candidates[best_candidate_idx]
    new_facilities_list.append([best_cand[0], best_cand[1]])
    
    # Update current minimum distances for all blocks
    current_min_dist = np.minimum(current_min_dist, best_candidate_dists)
    
    # Calculate metrics based on total patients
    covered_patients = ((current_min_dist <= R) * pb_counts).sum()
    cov = (covered_patients / len(patients)) * 100
    avg_d = (current_min_dist * pb_counts).sum() / len(patients)
    gain_from_base = cov - coverage_before
    
    # We need to map the grouped min_dist back to the full patients dataframe at the end
    # but for intermediate results we don't need it.
    
    results.append({
        "k": k, "coverage": cov, "gain_base": gain_from_base,
        "avg_dist": avg_d, "centers": np.array(new_facilities_list)
    })
    
    print(f"  {k:3d}    {cov:6.1f}%    +{gain_from_base:5.2f} pp    {avg_d:6.2f} km")
    
    if cov >= TARGET_COVERAGE:
        print(f"\n  >> Target coverage of {TARGET_COVERAGE}% reached!")
        break

optimal = results[-1]
K = optimal["k"]

new_facilities = pd.DataFrame(
    optimal["centers"],
    columns=["new_lat", "new_lon"]
)

# Apply final distances back to original patients dataframe for the map
final_centers = optimal["centers"]
patients["distance_after"] = patients["distance_to_nearest_provider"]
for center in final_centers:
    d = haversine_km(patients["pat_lat"].values, patients["pat_lon"].values, center[0], center[1])
    patients["distance_after"] = np.minimum(patients["distance_after"], d)
patients["is_covered_after"] = patients["distance_after"] <= R

print(f"\n  ** Optimal K = {K} **")
print("  Suggested new facility locations:")
for i, row in new_facilities.iterrows():
    print(f"    Facility {i+1}: ({row['new_lat']:.4f}, {row['new_lon']:.4f})")

# (distances and coverage already computed above)

coverage_after = patients["is_covered_after"].mean() * 100
avg_dist_after = patients["distance_after"].mean()

# ==========================================
# STEP 9 - OUTPUT METRICS
# ==========================================
print("\n" + "=" * 50)
print("          FACILITY LOCATION ANALYSIS")
print("=" * 50)
print(f"  Total patients analyzed:  {len(patients):,}")
print(f"  Coverage radius:          {R} km")
print(f"  New facilities suggested: {K}")
print(f"  --")
print(f"  Coverage BEFORE:          {coverage_before:.1f}%")
print(f"  Coverage AFTER:           {coverage_after:.1f}%")
print(f"  Improvement:              +{coverage_after - coverage_before:.1f} pp")
print(f"  --")
print(f"  Avg distance BEFORE:      {avg_dist_before:.2f} km")
print(f"  Avg distance AFTER:       {avg_dist_after:.2f} km")
print(f"  Distance reduction:       {avg_dist_before - avg_dist_after:.2f} km")
print("=" * 50)
print(f"\n  >> With {K} new facilities, coverage improves")
print(f"     from {coverage_before:.1f}% to {coverage_after:.1f}%")
print()

# ==========================================
# STEP 10 - COMPARISON ACROSS R VALUES
# ==========================================
print("--- Coverage comparison across radii ---")
for r_val in R_VALUES:
    cov_before = (patients["distance_to_nearest_provider"] <= r_val).mean() * 100
    cov_after = (patients["distance_after"] <= r_val).mean() * 100
    print(f"  R={r_val:2d} km:  {cov_before:5.1f}%  ->  {cov_after:5.1f}%  (+{cov_after - cov_before:.1f} pp)")
print()

# ==========================================
# STEP 11 - INTERACTIVE MAP
# ==========================================
print("Rendering map...")

# --- Helper: generate circle coordinates around a point ---
def circle_coords(lat_center, lon_center, radius_km, n_points=60):
    """Return lists of (lats, lons) forming a circle of radius_km around center."""
    R_EARTH = 6371.0
    lats = []
    lons = []
    for i in range(n_points + 1):  # +1 to close the circle
        angle = 2 * np.pi * i / n_points
        # Offset in radians
        dlat = (radius_km / R_EARTH) * np.cos(angle)
        dlon = (radius_km / R_EARTH) * np.sin(angle) / np.cos(np.radians(lat_center))
        lats.append(lat_center + np.degrees(dlat))
        lons.append(lon_center + np.degrees(dlon))
    return lats, lons

fig = go.Figure()

# --- Radar circles for EXISTING facilities (blue glow) ---
for idx, prow in providers.iterrows():
    clats, clons = circle_coords(prow["prov_lat"], prow["prov_lon"], R)
    fig.add_trace(go.Scattermap(
        lat=clats, lon=clons,
        mode="lines",
        fill="toself",
        fillcolor="rgba(52, 152, 219, 0.10)",
        line=dict(color="rgba(52, 152, 219, 0.35)", width=1),
        showlegend=False,
        hoverinfo="skip",
    ))

# --- Radar circles for NEW facilities (magenta glow) ---
for idx, nrow in new_facilities.iterrows():
    clats, clons = circle_coords(nrow["new_lat"], nrow["new_lon"], R)
    fig.add_trace(go.Scattermap(
        lat=clats, lon=clons,
        mode="lines",
        fill="toself",
        fillcolor="rgba(255, 0, 255, 0.10)",
        line=dict(color="rgba(255, 0, 255, 0.35)", width=1),
        showlegend=False,
        hoverinfo="skip",
    ))

# Calculate final coverage status for each block for the map
patient_blocks["is_covered_after"] = patient_blocks["min_dist"] <= R

# Scale bubble sizes based on exact patient count
# (min 4px, max 25px)
min_size = 4
max_size = 25
max_count = patient_blocks["count"].max()
patient_blocks["marker_size"] = min_size + (patient_blocks["count"] / max_count) * (max_size - min_size)

# --- Covered patients (green bubbles) ---
covered = patient_blocks[patient_blocks["is_covered_after"] == True]
fig.add_trace(go.Scattermap(
    lat=covered["pat_lat"],
    lon=covered["pat_lon"],
    mode="markers",
    marker=dict(size=covered["marker_size"], color="#2ecc71", opacity=0.4),
    name=f"Covered patients",
    customdata=covered[["count"]].values,
    hovertemplate="Covered patients: %{customdata[0]}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
))

# --- Underserved patients (red bubbles) ---
uncovered = patient_blocks[patient_blocks["is_covered_after"] == False]
fig.add_trace(go.Scattermap(
    lat=uncovered["pat_lat"],
    lon=uncovered["pat_lon"],
    mode="markers",
    marker=dict(size=uncovered["marker_size"], color="#e74c3c", opacity=0.5),
    name=f"Underserved patients",
    customdata=uncovered[["count"]].values,
    hovertemplate="Underserved patients: %{customdata[0]}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
))

# --- Existing providers (blue, larger) ---
fig.add_trace(go.Scattermap(
    lat=providers["prov_lat"],
    lon=providers["prov_lon"],
    mode="markers",
    marker=dict(size=14, color="#3498db", opacity=0.9),
    name=f"Existing facilities ({len(providers)})",
    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
    customdata=providers[["DepartmentName", "City"]].values,
))

# --- New suggested facilities (bright magenta, large) ---
fig.add_trace(go.Scattermap(
    lat=new_facilities["new_lat"],
    lon=new_facilities["new_lon"],
    mode="markers+text",
    marker=dict(size=22, color="#ff00ff", opacity=1.0),
    text=[f"NEW {i+1}" for i in range(K)],
    textposition="top center",
    textfont=dict(size=13, color="#ff00ff", family="Segoe UI, Arial"),
    name=f"NEW facilities ({K})",
    hovertemplate="<b>Suggested Facility %{text}</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
))

# --- Dummy trace for legend: coverage radius ---
fig.add_trace(go.Scattermap(
    lat=[None], lon=[None],
    mode="markers",
    marker=dict(size=12, color="rgba(255,255,255,0.15)"),
    name=f"Coverage radius ({R} km)",
    showlegend=True,
))

# --- Layout ---
center_lat = patients["pat_lat"].mean()
center_lon = patients["pat_lon"].mean()

fig.update_layout(
    title=dict(
        text=(
            f"<b>Healthcare Coverage Analysis</b>"
            f"<br><span style='font-size:13px;color:#8892b0'>"
            f"R = {R} km | {K} new facilities | "
            f"Coverage: {coverage_before:.1f}% -> {coverage_after:.1f}%</span>"
        ),
        font=dict(size=20, color="white", family="Segoe UI, Arial"),
        x=0.5, xanchor="center",
    ),
    font=dict(color="white", family="Segoe UI, Arial"),
    paper_bgcolor="#0f0f1a",
    map=dict(
        style="carto-darkmatter",
        center=dict(lat=center_lat, lon=center_lon),
        zoom=6,
    ),
    height=750,
    margin=dict(l=0, r=0, t=80, b=0),
    legend=dict(
        bgcolor="rgba(15,15,26,0.85)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
        font=dict(size=12, color="#ccc"),
        x=0.01, y=0.99,
        xanchor="left", yanchor="top",
    ),
)
import webbrowser, os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coverage_map.html")
fig.write_html(out_path)
webbrowser.open("file://" + out_path.replace("\\", "/"))
print(f"Map saved to: {out_path}")
print("Done!")
