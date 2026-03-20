# NYC Crime Spatiotemporal Prediction
### DS340W Capstone — Spring 2026

---

## 📌 Project Overview

This project adapts the **Informer + ST-GCN** framework from Fan et al. (2025) — originally validated on Chicago — to predict crime at the **police precinct level in New York City**. The goal is to forecast daily crime counts across NYC's 77 precincts using historical crime data, weather, and holiday features.

---

## 🔬 Research Question

> Can the Informer + ST-GCN framework, validated on Chicago, be successfully adapted to predict crime at the NYC police precinct level — and what modifications does the NYC context require?

---

## 📂 Dataset

| Field | Details |
|-------|---------|
| **Source** | [NYC Open Data — NYPD Complaint Data Historic](https://data.cityofnewyork.us/) |
| **File** | `NYPD_Complains_20060101_20241231_Clean` |
| **Time Range** | January 1, 2006 — December 31, 2024 |
| **Coverage** | 77 NYC Police Precincts |
| **External Features** | Weather (rp5.ru / NOAA), U.S. Holiday Calendar |

> 📁 Dataset is too large to host on GitHub directly. Access it here:
> **[Google Drive — Project Data Folder](https://drive.google.com/drive/u/0/folders/1JiRXj1JGuZAQ91Tq97K9XRDws_BA_Tg7)**
>
> Original source: [NYC Open Data — NYPD Complaint Data Historic](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i)

---

## 🗂️ Repository Structure

```
nyc-crime-spatiotemporal-prediction/
│
├── data/
│   ├── NYPD_Complains_20060101_20241231_Clean.csv   # Main dataset
│   └── weather/                                      # Weather data files
│
├── notebooks/
│   └── 01_data_cleaning.ipynb                        # Data preprocessing
│
├── src/
│   ├── preprocess.py                                 # Data pipeline scripts
│   ├── graph_construction.py                         # Precinct adjacency matrix
│   ├── model_stgcn.py                                # ST-GCN module
│   └── model_informer.py                             # Informer module
│
├── results/                                          # Outputs, figures, metrics
│
└── README.md
```

---

## 🧠 Methodology

This project follows the three-stage pipeline from the parent paper with NYC-specific adaptations:

1. **Data Preparation** — Clean and preprocess the NYPD dataset. Classify crimes into target categories. Merge with weather and holiday data.

2. **Graph Construction** — Model NYC's 77 precincts as nodes in a spatial graph. Build the adjacency matrix from official NYPD precinct shapefiles based on geographic contiguity.

3. **Informer + ST-GCN Model** — ST-GCN captures spatial crime patterns across neighboring precincts. Informer handles long-range temporal dependencies using ProbSparse self-attention. Features are fused via CNN and a fully connected output layer.

### Novel Contributions (vs. Parent Paper)
- **Finer spatial granularity**: 77 NYC precincts vs. 22 Chicago districts
- **STL decomposition**: Seasonal-trend decomposition applied before modeling to handle non-stationarity
- **Borough-level analysis**: Evaluating spatial error patterns across Manhattan, Brooklyn, the Bronx, Queens, and Staten Island
- **Broader temporal scope**: 2006–2024 dataset vs. the parent paper's 2015–2020

---

## 📄 Reference Papers

1. **Fan, Hu & Hu (2025)** — *Research on a Crime Spatiotemporal Prediction Method Integrating Informer and ST-GCN: A Case Study of Four Crime Types in Chicago.* Big Data and Cognitive Computing. *(Parent Paper)*

2. **Butt et al. (2025)** — *START: A Spatiotemporal Autoregressive Transformer for Enhancing Crime Prediction Accuracy.* IEEE Transactions on Computational Social Systems.

3. **Srivastava, Raj & Nithyashri (2025)** — *Dual-Attention Graph Neural Networks for SpatioTemporal Crime Forecasting and Patrol Optimization.* IEEE WAIE Workshop.

---

## 📊 Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

---

## ⚠️ Ethical Considerations

Predictive policing models trained on historical crime data can reflect systemic biases. This project uses only aggregate precinct-level counts and excludes individual-level sensitive attributes. Results should be interpreted as a decision-support tool, not an enforcement mechanism.

---

## 🚧 Status

- [x] Dataset acquired and cleaned
- [x] Preprocessing pipeline started
- [ ] Precinct adjacency matrix
- [ ] ST-GCN + Informer implementation
- [ ] Baseline model comparisons
- [ ] Final evaluation and analysis
