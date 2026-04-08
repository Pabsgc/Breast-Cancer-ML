# ⚖️ Feature Scaling in the ML Pipeline

## 📌 What is scaling?

Scaling is the process of:

> **Transforming all features to a comparable numerical scale**

---

## 🧠 Why does scaling matter?

Imagine your features look like this:

* `pitch` → values around 100–300
* `jitter` → values around 0.001
* `duration` → values around 10

When you feed this into a model:

👉 The model will **pay more attention to larger values**

### 💡 Key intuition

> A feature with larger magnitude can dominate the model, even if it is not more important.

---

## 🎯 What does scaling do?

Scaling transforms features so they are on a similar scale.

### 📊 StandardScaler

The most commonly used method:

* Mean = 0
* Standard deviation = 1

---

## 🧪 Example

### Before scaling:

```
pitch:     120, 150, 180
jitter:    0.002, 0.003, 0.001
```

### After scaling:

```
pitch:     -1.2, 0.0, 1.2
jitter:    -0.5, 1.0, -0.5
```

👉 Now all features are **comparable and balanced**

---

## ⚠️ Is scaling needed in this project?

### 🌲 Random Forest

* ❌ Does NOT require scaling
* Tree-based models are scale-invariant

---

### 📈 Logistic Regression

* ✅ DOES require scaling
* Improves:

  * convergence
  * numerical stability
  * interpretability

---

### 💡 Key takeaway for this pipeline

> If you are using Logistic Regression (which you should for interpretability), scaling is mandatory.

---

## 🔥 Common mistake (avoid this!)

Do **NOT** do this:

```python
scaler.fit(X_total)
```

before splitting into train/test.

💀 This causes **data leakage**

---

## ✅ Correct approach

Use a `Pipeline` from sklearn:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

👉 This ensures scaling is fitted **only on training data**

---

## 🧠 Pro-level insight (important for TFG)

In your project, scaling should be applied:

### ✅ AFTER:

* eGeMAPS feature extraction
* Feature merging (acoustic + NLP)

### ✅ BEFORE:

* Feature selection (if method is scale-sensitive)
* Logistic Regression

---

## ⚠️ Important nuance

If your pipeline is:

```
Random Forest → Feature Selection → Logistic Regression
```

Then:

* 🌲 Random Forest → NO scaling needed
* 📈 Logistic Regression → scaling REQUIRED

---

## 🧭 Clean summary

* Scaling = feature normalization
* Required for linear models
* Not required for tree-based models
* Must be inside a pipeline to avoid data leakage

---

## 🎯 Final insight

In this project:

> Scaling is not a minor detail — it is what makes your **interpretable model (Logistic Regression)** work correctly.

---

## 🚀 Next steps

Future improvements to implement:

* Full sklearn pipeline including:

  * Feature selection
  * Random Forest
  * Logistic Regression
  * Proper cross-validation

👉 This moves the project from a prototype to a **research-grade system**
