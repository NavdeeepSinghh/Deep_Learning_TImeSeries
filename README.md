# 📊 Time Series Forecasting Pipeline (Custom GRU/RNN)

This project implements a **complete end-to-end time-series forecasting pipeline from scratch**, with a strong focus on *understanding* rather than just execution.

The goal is to deeply explore:

* How sequential data is transformed and fed into models
* How neural networks utilize past information
* Where these models fail and why those failures occur

---

# 🧠 Objective

Unlike standard implementations, this project emphasizes **conceptual clarity**:

* Understanding how time-series differs from traditional tabular data
* Observing how models process sequential dependencies
* Analyzing model limitations through real outputs

> The core idea is not just *"does it work?"* but *"why does it work, and when does it fail?"*

---

# ⚙️ Personalized Parameter Setup

All model configurations are dynamically generated using the roll number:

```python
window_size = (sum of all digits) % 10 + 8
prediction_horizon = (last 2 digits) % 3 + 1
hidden_size = (first 3 digits) % 16 + 8
```

### 🔍 Why this matters:

* Ensures **unique configurations per student**
* Demonstrates ability to build **dynamic pipelines**
* Prevents hardcoding and encourages generalization

---

# 📊 Datasets Used

## 1. ⚡ Electricity Consumption Dataset

* Structured and periodic data
* Exhibits clear trends and seasonal patterns
* Easier for sequence models to learn

## 2. 📉 Stock Price Dataset (AAPL)

* Real-world financial dataset
* Highly volatile and noisy
* Weak temporal patterns

### 🔍 Insight:

> Comparing these datasets highlights how **data characteristics affect model performance**

---

# 🔄 Data Preprocessing Pipeline

## ✔ Normalization

* MinMax Scaling applied
* Converts values into range [0,1]

### Why?

* Prevents large values from dominating learning
* Improves training stability

---

## 🔁 Sliding Window Transformation (Core Concept)

Time-series data is not directly usable by neural networks.
It must be converted into a **supervised learning format**.

### Process:

```text
Original Data: [x1, x2, x3, x4, x5, x6]

Window Size = 3

Input  → [x1, x2, x3] → Output → [x4]  
Input  → [x2, x3, x4] → Output → [x5]
```

### 🔍 Why this works:

* Provides **temporal context**
* Allows model to learn dependencies from past values

---

# 🧱 Models Implemented

## 🔹 1. MLP (Baseline Model)

* Treats input as a flat vector
* No understanding of sequence order

### Limitation:

> Cannot capture temporal relationships

---

## 🔹 2. Custom GRU (Implemented from Scratch)

Built manually without using `nn.GRU`.

### Internal Mechanism:

* **Update Gate (z)** → controls memory retention
* **Reset Gate (r)** → controls forgetting
* **Hidden State (h)** → carries temporal information

### Key Idea:

> The GRU learns *what to remember and what to forget* from past data

---

## 🔹 3. LSTM (Prebuilt for Comparison)

* More complex memory structure
* Handles long-term dependencies better than GRU

---

## 🔹 4. Transformer Model

* Uses **attention mechanism**
* Captures global relationships instead of step-by-step memory

### Key Advantage:

> Can model long-range dependencies more effectively

---

# 🏋️ Training Strategy

* Chronological split (no shuffling)
* Ensures real-world simulation of forecasting
* Loss Function: Mean Squared Error (MSE)
* Optimizer: Adam

---

# 📏 Evaluation Metrics

| Metric | Meaning                         |
| ------ | ------------------------------- |
| MSE    | Penalizes large errors          |
| MAE    | Measures average absolute error |
| RMSE   | Interpretable error scale       |

---

# 📊 Results & Observations

## ⚡ Electricity Dataset

* RMSE ≈ **0.40**
* Model successfully captures trends

### Interpretation:

> Data has strong patterns → easier to learn

---

## 📉 Stock Dataset

* RMSE ≈ **0.77**
* Significant performance drop

### Interpretation:

> High volatility → weak learnable structure

---

# 🧪 Ablation Study (Window Size Impact)

| Window Size | Observation                         |
| ----------- | ----------------------------------- |
| Small       | Underfitting (insufficient context) |
| Medium      | Balanced performance                |
| Large       | More context but increased noise    |

### Insight:

> Window size directly controls how much history the model can access

---

# 💥 Failure Analysis (Critical Section)

## ❌ 1. Poor Performance on Volatile Data

* Stock prices lack consistent patterns
* Model struggles to generalize

---

## ❌ 2. Smoothing Effect

* Predictions are less sharp than actual values

### Reason:

> Model minimizes error → avoids extreme predictions

---

## ❌ 3. Failure to Capture Spikes

* Sudden changes are rare → not learned effectively

---

## ❌ 4. Limited Memory

* GRU struggles with long-term dependencies

---

# 🔍 Key Learnings

* Sequence models outperform MLP due to temporal awareness
* Model performance depends heavily on data structure
* More context ≠ always better
* Understanding failure is essential in deep learning

---

# 📌 Conclusion

This project demonstrates that:

* Time-series forecasting requires careful data transformation
* Sequence models provide significant advantages over static models
* However, they are limited by:

  * data quality
  * noise
  * model architecture

---

# 🚀 Future Improvements

* Use attention-based models for better long-range learning
* Hyperparameter tuning
* Multi-step forecasting
* Larger datasets

---

# 🙌 Author

**Navdeep Singh**
B.Tech Computer Science
