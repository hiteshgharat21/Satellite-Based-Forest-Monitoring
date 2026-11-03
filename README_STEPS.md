# Uttarakhand Forest Health & Fire Risk — Step-by-Step Guide

## STEP 0: Setup (30 minutes)

1. Create a Google Earth Engine account: https://earthengine.google.com → Register (free for eligible student/non-commercial use).
2. Go to console.cloud.google.com and create a project. Copy your Project ID and enable the Earth Engine API.
3. Use Google Colab (free GPU, no installation required) or install the following packages on your laptop:

   `pip install earthengine-api xgboost shap pandas scikit-learn`

4. In Terminal or Colab, run:

   `earthengine authenticate`

## STEP 1: Data Export (`01_gee_export/export_features.py`)

- Enter your Project ID in `GEE_PROJECT` and run the script.
- Go to `code.earthengine.google.com/tasks` and click RUN on all 7 tasks (2019–2025).
- After 10–40 minutes, CSV files will appear in `Drive/uk_forest`. Copy them into the `02_data/` folder.
- **CHECK:** `print("Forest tiles")` should return approximately 300–500. NDVI values in the CSV should generally range between 0.2 and 0.9.

## STEP 2: Merge & Label (`03_model/merge_and_label.py`)

- Combines all 7 CSV files.
- Calculates changes in NDVI, NDMI, and NBR, along with their rolling averages.
- **LABEL:** Determines whether a fire occurred in the following month using MODIS burned-area data.
- **Data split:** Training: 2019–2023, Validation: 2024, Testing: 2025 (time-based split, not random).

## STEP 3: XGBoost Baseline + SHAP (`03_model/train_xgboost.py`)

- Start with tabular features only.
- Evaluate ROC-AUC and PR-AUC.
- Generate `shap_summary.png` to visualize the Explainable AI (XAI) results.
- This will be your first actual AI result.

## STEP 4: Vision Transformer (ViT) — Next Stage

- Export 224 × 224 Sentinel-2 image chips for each tile-month from Google Earth Engine.
- Use a pretrained Vision Transformer (ViT) from `timm` to extract 768-dimensional features.
- Apply PCA and combine the extracted features with the XGBoost features.
- Compare the results of XGBoost alone against the combined ViT + XGBoost model. **This comparison represents the novelty of your project.**
- Visualize the attention maps.

## STEP 5: Dashboard

- Develop an interactive map using Leaflet or Folium.
- Display tile polygons, NDVI, NDMI, NBR, colour-coded fire-risk levels, and the top contributing factors identified by SHAP.

## STEP 6: Scale to India

- In the Uttarakhand code, change only the boundary filter (use FAO GAUL Level 0 for India or a state-wise processing loop).
- Keep the same tile grid, features, and model.
- More cloud computing resources will be required, so export the data state by state.
