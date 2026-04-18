# NYC Crime Spatiotemporal Prediction
### DS340W Capstone Project — Group 60 — Spring 2026

## Project Overview

This project adapts the **Informer + ST-GCN** hybrid model from Fan et al. (2025) to predict crime at the police precinct level in New York City. The original model was developed and tested on Chicago crime data. This study extends it to NYC to evaluate cross-city generalization and adds **STL decomposition** as a novel preprocessing enhancement.

**Author:** Redza Nisham (mkn5378@psu.edu)  
**Course:** DS340W, Penn State University

## Research Question

Can the Informer + ST-GCN framework, originally validated on Chicago, be successfully adapted to predict crime at the NYC police precinct level, and does STL decomposition improve prediction accuracy?

## Key Results

| Model | Assault MAE | Robbery MAE | Crim. Damage MAE | Theft MAE |
|-------|------------|------------|------------------|-----------|
| Linear Regression | 1.53 | 0.60 | 1.45 | 1.80 |
| Ridge Regression | 1.53 | 0.60 | 1.45 | 1.80 |
| Random Forest | 1.58 | 0.62 | 1.72 | 1.83 |
| LSTM | 2.47 | 0.62 | 1.83 | 4.41 |
| Informer+ST-GCN | 1.63 | 0.61 | 1.51 | 2.06 |
| **Informer+ST-GCN + STL (Ours)** | **1.64** | **0.60** | **1.52** | **1.94** |

The STL enhanced model reduces theft prediction error by **6.1%** and training loss by **20%** compared to the base model.

## Dataset

| Field | Details |
|-------|---------|
| **Crime Source** | NYC Open Data Portal — NYPD Complaint Data Historic |
| **Weather Source** | Open-Meteo Historical Weather API (Central Park station) |
| **Time Range** | January 1, 2015 to March 10, 2020 |
| **Crime Records** | 938,414 (after filtering) |
| **Precincts** | 49 active NYPD precincts |
| **Crime Types** | Assault, Robbery, Criminal Damage, Theft |
| **External Features** | Temperature, humidity, wind speed, precipitation, holidays |

The processed dataset is too large for GitHub. Raw data can be downloaded by running Notebook 1.

## Repository Structure

```
nyc-crime-prediction/
│
├── notebooks/
│   ├── Notebook_1_Data_Collection_and_Preprocessing.ipynb
│   ├── Notebook_2_EDA.ipynb
│   ├── Notebook_3_Model_Implementation.ipynb
│   └── Notebook_4_Baselines_and_Evaluation.ipynb
│
├── outputs/
│   ├── crime_over_time.png
│   ├── crime_by_type.png
│   ├── monthly_seasonality.png
│   ├── day_of_week.png
│   ├── top_precincts.png
│   ├── correlation_heatmap.png
│   ├── adjacency_graph.png
│   ├── training_loss.png
│   ├── mae_by_precinct.png
│   ├── timeseries_assault.png
│   ├── timeseries_criminal_damage.png
│   ├── timeseries_robbery.png
│   ├── timeseries_theft.png
│   ├── barchart_assault.png
│   ├── barchart_criminal_damage.png
│   ├── barchart_robbery.png
│   ├── barchart_theft.png
│   ├── model_comparison_results.csv
│   └── final_comparison_with_stl.csv
│
├── paper/
│   └── Final_Paper.docx
│
├── README.md
└── requirements.txt
```

## How to Run

All notebooks are designed to run in **Google Colab** (free tier with GPU).

1. Open each notebook in Google Colab
2. Run cells from top to bottom
3. Notebook 1 must be run first (downloads and processes all data)
4. Each notebook saves results to Google Drive, so progress is preserved if Colab disconnects

**Runtime requirement:** Set runtime to **GPU** (Runtime > Change runtime type > T4 GPU) for Notebook 3.

### Notebook Descriptions

| Notebook | Purpose | Time |
|----------|---------|------|
| Notebook 1 | Download NYC crime, weather, holiday data. Clean, merge, build adjacency matrix, train/test split. | ~15 min |
| Notebook 2 | Exploratory data analysis. Generate all visualizations. | ~2 min |
| Notebook 3 | Build and train the Informer + ST-GCN model. Run STL enhancement. | ~20 min |
| Notebook 4 | Run baseline models (Linear, Ridge, RF, LSTM). Compute metrics. Generate result figures. | ~10 min |

## Model Architecture

The hybrid model has three modules:

1. **ST-GCN Module** — Captures spatial dependencies between precincts using graph convolutions on the adjacency matrix. Processes proximity, periodic, and trend features through separate residual GCN branches.

2. **Informer Module** — Captures temporal patterns using transformer self-attention with convolutional distillation. Shared encoder processes all precincts in parallel.

3. **Fusion Layer** — Concatenates ST-GCN and Informer outputs, maps through fully connected layers to predicted crime counts.

**Novel Enhancement:** STL decomposition removes weekly seasonality before model input, reducing theft MAE by 6.1%.

**Parameters:** 239,492 | **Optimizer:** NAdam | **Lookback:** 7 days | **Test period:** Jan 1-7, 2020

## Dependencies

```
torch
numpy
pandas
scikit-learn
matplotlib
seaborn
geopandas
sodapy
openmeteo-requests
requests-cache
retry-requests
statsmodels
networkx
```

## Reference Papers

1. **Fan, Hu & Hu (2025)** — *Research on a Crime Spatiotemporal Prediction Method Integrating Informer and ST-GCN.* Big Data and Cognitive Computing. **(Parent Paper)**
2. **Butt et al. (2025)** — *START: A Spatiotemporal Autoregressive Transformer for Enhancing Crime Prediction Accuracy.* IEEE Trans. Computational Social Systems.
3. **Srivastava, Raj & Nithyashri (2025)** — *Dual-Attention Graph Neural Networks for SpatioTemporal Crime Forecasting.* IEEE WAIE Workshop.

## Ethical Considerations

Predictive policing models trained on historical crime data can reflect systemic biases. This project uses only aggregate precinct-level counts and excludes individual-level sensitive attributes. Results should be interpreted as a decision-support tool, not an enforcement mechanism.
