# Datafest 2026 - Diabetes Patient Journey UMAP Analysis

This project analyzes diabetes patient journey types using UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction and visualization.

## Project Structure

```
Datafest/
├── umap_journey_types.py          # Main UMAP training and visualization script
├── umap_journey_contours.py       # UMAP visualization adjustment script (no retraining)
├── journey_umap_patients.csv      # Patient features with UMAP coordinates
├── team21_journey_umap.pkl        # Trained UMAP model
├── team21_journey_scaler.pkl      # Feature scaler for UMAP
└── output_journey_umap/           # Generated outputs (not in git)
```

## Scripts

### umap_journey_types.py
Main script for training UMAP model and generating visualizations.

**Features:**
- Filters diabetes patients (E10, E11, E13, E14 ICD-10 codes)
- Calculates patient journey features (velocity, redundancy, latency, diversity)
- Trains supervised UMAP with journey type labels
- Generates visualization with KDE-based cluster shading
- Saves UMAP model and scaler for reuse

**Usage:**
```bash
python umap_journey_types.py
```

**Outputs:**
- `output_journey_umap/umap_journey_types.png` - UMAP visualization
- `output_journey_umap/journey_umap_patients.csv` - Patient features with UMAP coordinates
- `team21_journey_umap.pkl` - Trained UMAP model
- `team21_journey_scaler.pkl` - Feature scaler

### umap_journey_contours.py
Script for adjusting UMAP visualization parameters without retraining the model.

**Features:**
- Loads existing UMAP data and model
- Adjusts cluster shading parameters (alpha, start_level, bandwidth)
- Updates visualization without re-running UMAP training
- Useful for iterative refinement of visualization appearance

**Usage:**
```bash
python umap_journey_contours.py
```

**Adjustable Parameters:**
- `start_level`: Controls which contour levels to shade (higher = fewer levels shaded)
- `alpha`: Shading transparency (lower = more transparent)
- `bandwidth`: KDE bandwidth for contour smoothing

## Journey Types

The analysis classifies patients into three journey types:

1. **Abandoned (Static Journey)** - Patients who disengage from care
2. **Cycler (Clinical Redirection)** - Patients who cycle through different care pathways
3. **Escalator (Care Escalation)** - Patients whose care escalates over time

## Visualization Features

- **KDE-based cluster shading** - Transparent shading around clusters using kernel density estimation
- **Color coding** - Red (Abandoned), Yellow (Cycler), Green (Escalator)
- **Legend** - Positioned outside the plot for clarity
- **Axis labels** - UMAP_1 and UMAP_2

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- umap-learn
- matplotlib
- seaborn
- scipy

## Installation

```bash
pip install pandas numpy scikit-learn umap-learn matplotlib seaborn scipy
```

## Notes

- The UMAP model is trained on ~27k diabetes patients
- Random state is set to 42 for reproducibility
- The visualization uses refined shading parameters for optimal cluster visibility
