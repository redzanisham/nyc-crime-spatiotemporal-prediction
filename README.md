# NYC Crime Spatiotemporal Prediction

**Hybrid Informer + ST-GCN model with STL decomposition for predicting crime in New York City**

DS340W Capstone Project | Penn State University | Spring 2026

---

## Author

**Redza Nisham**  
Department of Data Sciences, Penn State University  
mkn5378@psu.edu

---

## Project Summary

This project adapts the hybrid Informer + Spatiotemporal Graph Convolutional Network (ST-GCN) model proposed by Fan et al. (2025) for crime prediction in Chicago, applying it to New York City and adding STL decomposition as a novel preprocessing enhancement.

Using approximately 938,000 NYPD crime records across 49 precincts from 2015 to 2020, the model predicts daily counts of four crime types: assault, robbery, criminal damage, and theft. The STL decomposition preprocessing reduces theft prediction error by 6.1% and improves training convergence by 20% compared to the base model.

## Key Results

| Model | Assault MAE | Robbery MAE | Crim. Damage MAE | Theft MAE |
|-------|------------|------------|------------------|-----------|
| Linear Regression | 1.53 | 0.60 | 1.45 | 1.80 |
| Ridge Regression | 1.53 | 0.60 | 1.45 | 1.80 |
| Random Forest | 1.58 | 0.62 | 1.72 | 1.83 |
| LSTM | 2.47 | 0.62 | 1.83 | 4.41 |
| Informer + ST-GCN | 1.63 | 0.61 | 1.51 | 2.06 |
| **Informer + ST-GCN + STL (Ours)** | **1.64** | **0.60** | **1.52** | **1.94** |

The model beats the LSTM baseline by 34-56% across all four crime types.

## Repository Structure

```
nyc-crime-spatiotemporal-prediction/
│
├── notebooks/
│   ├── Notebook_1_Data_Collection_and_Preprocessing.ipynb   # Data pipeline
│   ├── Notebook_2_EDA.ipynb                                 # Visualizations
│   ├── Notebook_3_Model_Implementation.ipynb                # Model + STL training
│   └── Notebook_4_Baselines_and_Evaluation.ipynb            # Baseline comparison
│
├── outputs/                  # Generated figures and CSVs
│   ├── crime_over_time.png
│   ├── crime_by_type.png
│   ├── monthly_seasonality.png
│   ├── day_of_week.png
│   ├── top_precincts.png
│   ├── correlation_heatmap.png
│   ├── adjacency_graph.png
│   ├── training_loss.png
│   ├── mae_by_precinct.png
│   ├── timeseries_*.png
│   ├── barchart_*.png
│   ├── model_comparison_results.csv
│   └── final_comparison_with_stl.csv
│
├── paper/                    # Final research paper
│   └── Final_Paper.docx
│
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

All notebooks are designed for **Google Colab** with GPU runtime (T4).

### Setup
1. Upload notebooks to Google Drive in a `capstone_project/notebooks/` folder
2. Open each notebook in Google Colab
3. Set runtime to **GPU**: Runtime → Change runtime type → T4 GPU
4. Run cells in order from top to bottom

### Run Order
| Notebook | Purpose | Approximate Runtime |
|----------|---------|---------------------|
| Notebook 1 | Download and preprocess all data | 15 minutes |
| Notebook 2 | Generate EDA visualizations | 2 minutes |
| Notebook 3 | Train Informer + ST-GCN, then STL enhanced version | 25 minutes |
| Notebook 4 | Train baselines and compute final metrics | 10 minutes |

Each notebook saves intermediate results to Google Drive, so progress is preserved if Colab disconnects.

## Method Overview

The model has three main modules:

### 1. ST-GCN Module (Spatial)
Treats the 49 NYC precincts as a graph where edges connect precincts that share a border. Applies graph convolutions following Kipf and Welling's GCN formulation. Processes three branches in parallel: proximity (recent days), periodic (weekly patterns), and trend (longer term shifts).

### 2. Informer Module (Temporal)
Uses transformer self-attention to capture temporal patterns in each precinct's time series. Includes ProbSparse self-attention and convolutional distillation for efficiency.

### 3. STL Decomposition (Novel Enhancement)
Splits each precinct's crime time series into trend, seasonal, and residual components using STL with period=7. The model receives only the trend + residual, removing predictable weekly patterns so it can focus on harder predictions. Inspired by Butt et al.'s START framework.

The outputs from ST-GCN and Informer are concatenated through a fusion layer to produce final crime predictions.

**Model parameters:** 239,492  
**Training:** 2015-01-01 to 2019-12-31 (1,826 days)  
**Test:** 2020-01-01 to 2020-01-07 (7 days)  
**Optimizer:** NAdam with cosine annealing (lr=0.001)  
**Lookback window:** 7 days

## Dataset

| Source | Details |
|--------|---------|
| **Crime data** | NYC Open Data Portal — NYPD Complaint Data Historic |
| **Weather** | Open-Meteo Historical Weather API (Central Park station) |
| **Adjacency** | NYC ArcGIS Police Precinct Shapefiles |
| **Time range** | January 1, 2015 to March 10, 2020 |
| **Records** | 938,414 (after filtering and cleaning) |
| **Precincts** | 49 active NYPD precincts |
| **Features** | Crime counts, weather, holidays, derived weather metrics, rolling averages |

## Reference Papers

1. **Fan, Y., Hu, X., & Hu, J. (2025)** — *Research on a Crime Spatiotemporal Prediction Method Integrating Informer and ST-GCN: A Case Study of Four Crime Types in Chicago.* Big Data and Cognitive Computing, 9(179). **(Parent paper)**

2. **Butt, U. M., Letchmunan, S., Ali, M., & Sherazi, H. H. R. (2025)** — *START: A Spatiotemporal Autoregressive Transformer for Enhancing Crime Prediction Accuracy.* IEEE Transactions on Computational Social Systems, 12(6), 4650-4664.

3. **Srivastava, M., Raj, S., & Nithyashri, J. (2025)** — *Dual-Attention Graph Neural Networks for SpatioTemporal Crime Forecasting and Patrol Optimization.* Proc. 7th International Workshop on AI and Education, 416-420.

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
shapely
```

Install with:
```bash
pip install -r requirements.txt
```

## Ethical Considerations

Predictive policing models trained on historical crime data can reflect and amplify systemic biases in past law enforcement practices. This project uses only aggregate precinct-level counts and excludes individual-level sensitive attributes.

**This work should be interpreted as a decision-support tool for resource allocation research, not as an operational enforcement mechanism.** Any practical deployment would require careful auditing for bias, regular validation against ground truth, and human oversight of any decisions influenced by the predictions.

## License

This is an academic capstone project. The code is provided for educational and research purposes.

## Contact

Questions about this work? Reach out at mkn5378@psu.edu
