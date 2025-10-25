# CSI300 Stock Price Forecasting using LSTM and Transformer

## Goal

The goal of this project is to build and compare **LSTM** and **Transformer** deep learning models for predicting the **Close, Open, Low, and High** prices of the CSI300 index.  

Through this comparison, the project aims to:
- Evaluate the ability of **attention-based architectures** to model nonlinear temporal dependencies in financial time series;  
- Demonstrate how **deep learning approaches** can improve upon traditional sequential models in market forecasting applications.

---

## Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Methods](#2-methods)
  - [2.1 Dataset](#21-dataset)
  - [2.2 Preprocessing](#22-preprocessing)
  - [2.3 Models](#23-models)
- [3. Results](#3-results)
- [4. Insights](#4-insights)
- [5. Environment & Dependencies](#5-environment--dependencies)
- [6. Project Structure](#6-project-structure)
- [7. Future Work](#7-future-work)
- [8. Author](#8-author)

---

## 1. Project Overview

Financial time series data, such as stock prices, exhibit complex **nonlinear and long-term dependencies**.  
Traditional models often fail to capture these dynamics, especially under volatile market conditions.  

This project applies two deep learning architectures — **LSTM** and **Transformer** — to predict multiple stock price variables for the **CSI300 index**.  
By comparing their predictive performance, we highlight how attention mechanisms enhance forecasting accuracy and stability.

---

## 2. Methods

### 2.1 Dataset
- Historical daily price data for the **CSI300 index**, containing columns:  
  `trade_date`, `close`, `open`, `low`, `high`.  
- Time range: 2013–2024.  
- Data split:  
  - **Training set:** before 2022-01-01  
  - **Testing set:** from 2022-01-01 onward  

### 2.2 Preprocessing
- Converted `trade_date` to datetime format and sorted the dataset chronologically.  
- Normalized price columns using `MinMaxScaler` to scale all features into [0, 1].  
- Constructed supervised sequences with a **window size of 60 days** (`sequence_length = 60`), where each input sequence predicts the next day’s four price values.  
  - A sequence length of 60 was chosen empirically, balancing short-term sensitivity and long-term trend stability.

### 2.3 Models

#### LSTM
- **Architecture:**  
  - Two stacked LSTM layers (50 units each) with dropout (0.2)  
  - Dense(25) → Dense(4) output layer for 4 price predictions  
- **Loss:** Mean Squared Error  
- **Optimizer:** Adam (`lr = 0.001`)  
- **Epochs:** 50  
- **Batch size:** 32  

LSTM captures temporal dependencies through gated recurrent units, effectively modeling smooth sequential patterns in price data.

#### Transformer
- **Architecture:**  
  - 2 encoder and 2 decoder layers  
  - `d_model = 128`, `nhead = 8`, `dropout = 0.1`  
  - Feed-forward network dimension: 256  
- **Loss:** MSE  
- **Optimizer:** Adam (`lr = 1e-4`) with `ReduceLROnPlateau` scheduler  
- **Regularization:** Gradient clipping (max_norm = 1.0)  
- **Epochs:** 50  

Transformers leverage **self-attention mechanisms** to learn global temporal relationships, handling long-range dependencies better than LSTM.

---

## 3. Results

| Target | Model | MSE | RMSE | MAE | R² |
|--------|--------|------|------|------|------|
| Close | LSTM | 3516.14 | 59.30 | 48.16 | 0.9240 |
| Close | Transformer | 3880.78 | 62.30 | 49.50 | 0.9161 |
| Open | LSTM | 2226.17 | 47.18 | 41.19 | 0.9522 |
| Open | Transformer | 1785.07 | 42.25 | 34.68 | 0.9617 |
| Low | LSTM | 2148.66 | 46.35 | 35.55 | 0.9526 |
| Low | Transformer | 2507.69 | 50.08 | 39.96 | 0.9447 |
| High | LSTM | 2648.82 | 51.47 | 44.92 | 0.9421 |
| High | Transformer | 2406.51 | 49.06 | 38.81 | 0.9474 |

**Key findings:**
- Both models achieve **R² > 0.91**, demonstrating strong predictive power.  
- LSTM performs slightly better for *Close* and *Low* prices.  
- Transformer performs better for *Open* and *High* prices.  
- The Transformer shows smoother convergence and better generalization under longer sequence windows.

---

## 4. Insights

- The **Transformer’s attention mechanism** helps capture global temporal relationships beyond LSTM’s short memory horizon.  
- **LSTM** retains an advantage for stable, locally correlated series (e.g., *Close* price).  
- Both architectures are valuable in market forecasting, with complementary strengths — LSTM for short-term stability, Transformer for long-term trend detection.

---

## 5. Environment & Dependencies

**Python Version:** 3.10  

**Required Libraries:**
```
tensorflow
torch
numpy
pandas
matplotlib
scikit-learn
```

---

## 6. Project Structure
```text
csi300-prediction/
│
├── data/
│   └── csi300.csv
│
├── models/
│   ├── lstm_csi300_model.keras
│   └── transformer_csi300_model.pth
│
├── notebook/
│   └── CSI300_LSTM_Transformer.ipynb
│
├── results/
│   ├── lstm_prediction.png
│   ├── transformer_prediction.png
│   └── metrics_comparison.csv
│
└── README.md
```

---

## 7. Future Work

- Extend prediction to include **volatility** or **market indicators** (volume, turnover).  
- Incorporate **economic and sentiment features** for multivariate forecasting.  
- Visualize Transformer **attention weights** for interpretability.  
- Explore **ensemble or hybrid models** combining LSTM and Transformer outputs.  

---

## 8. Author

**Han Zheng**  
Master of Information, Human-Centered Data Science  
University of Toronto  
Bachelor of Science in Statistics, The O
