# 🧠 ML Learning Journey (PyTorch • Colab • From Scratch)

> A hands-on Machine Learning repository focused on **learning by building models from scratch** using PyTorch and Google Colab.




---

## 🚀 Overview

This repository documents my journey of learning Machine Learning through **practical implementation and experimentation**.

Instead of relying only on high-level APIs, I focus on:

* understanding how models work internally
* building training loops from scratch
* debugging real-world errors
* developing strong ML intuition


---

## 📘 Topics Covered

### 🔹 Binary Classification

* logits vs probabilities
* sigmoid function
* BCEWithLogitsLoss

### 🔹 Multiclass Classification

* softmax intuition
* CrossEntropyLoss
* class predictions using argmax

### 🔹 Neural Networks

* Linear layers
* Activation functions (ReLU)
* Model architecture design

### 🔹 Training Pipeline

* forward pass
* loss computation
* backpropagation
* optimizer step

### 🔹 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Custom metric implementation

---

## 🧪 Example Training Workflow

```python
# Forward pass
logits = model(X_train)

# Loss (multiclass)
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, y_train)

# Backpropagation
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 🚀 Featured Project

### 🔹 Classification on Moons Dataset

* Built a neural network from scratch using PyTorch
* Implemented full training loop manually
* Visualized decision boundaries
* Experimented with different architectures and hyperparameters


---

## ⚠️ Debugging & Learnings

Common issues I explored and solved:

* Shape mismatch in loss functions
* Incorrect use of softmax with CrossEntropyLoss
* Difference between logits and probabilities
* Device mismatch (CPU vs CUDA)
* Binary vs multiclass loss confusion

---

## 📌 Progress Tracker

* [x] Binary Classification
* [x] Multiclass Classification
* [x] Custom Metrics Implementation
* [ ] Computer Vision (CNNs)
* [ ] Model Deployment

---

## 🔜 Coming Soon

* 🖼️ Computer Vision (CNNs)
* 📷 Image classification projects
* 📈 Training visualizations (loss & accuracy curves)
* 🚀 More advanced architectures

---

## 🛠️ Tech Stack

* Python 🐍
* PyTorch 🔥
* NumPy
* Matplotlib
* Pandas
* Scikit-learn

---

## 💡 Goal

To build a strong foundation in Machine Learning and Deep Learning by:

* understanding concepts from first principles
* implementing models manually
* practicing consistently through experiments

---
