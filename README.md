# 🧠 ML Learning Journey (PyTorch • Colab • From Scratch)

> A hands-on ML repository focused on **building models from scratch** using PyTorch and Google Colab.

---

## 🚀 Overview

Instead of relying on high-level APIs, I focus on understanding models internally, building training loops from scratch, and developing strong ML intuition through experimentation.

---

## 📘 Topics Covered

| Area | Key Concepts |
|---|---|
| Binary Classification | Sigmoid, BCEWithLogitsLoss, logits vs probabilities |
| Multiclass Classification | Softmax, CrossEntropyLoss, argmax predictions |
| Neural Networks | Linear layers, ReLU, architecture design |
| Training Pipeline | Forward pass, backprop, optimizer step |
| Evaluation Metrics | Accuracy, Precision, Recall, F1-score |
| Computer Vision (CNNs) | Conv2d, MaxPool2d, TinyVGG, FashionMNIST |
| Model Persistence | `state_dict()` save & load, `.pth` files |

---

## 🖼️ FashionMNIST — Computer Vision

Trained and compared **3 architectures** on FashionMNIST, from a linear baseline to a full CNN.

| Model | Loss | Accuracy | Time |
|---|---|---|---|
| V0 — Baseline Linear | 0.4798 | 83.41% | 28.68s |
| V1 — Linear + ReLU | 0.6850 | 75.02% | 31.95s |
| **V2 — TinyVGG CNN** ✅ | **0.3273** | **88.34%** | 37.24s |

> 💡 The CNN outperforms both linear models — convolutional layers are far better suited for image data.


#### 🔀 Confusion Matrix
<img width="662" height="650" alt="image" src="https://github.com/user-attachments/assets/11e04752-5262-460a-95f6-3e0854ddd48c" />


```python
confmat = ConfusionMatrix(num_classes=len(class_name), task="multiclass")
confmat_tensor = confmat(preds=y_pred_tensor, target=test_data.targets)
fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(), class_names=class_name)
```

#### 💾 Model Saving & Loading
```python
# Save
torch.save(obj=model_2.state_dict(), f="models/03_pytorch_computer_vision_model_2.pth")

# Load
loaded_model_2 = FashionMNISTModelV2(input_shape=1, hidden_units=10, output_shape=10)
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model_2 = loaded_model_2.to(device)
```

> Saving `state_dict()` only keeps the learned weights — more portable than saving the full model object.

---

## 📌 Progress Tracker

* [x] Binary Classification
* [x] Multiclass Classification
* [x] Custom Metrics
* [x] Computer Vision (CNNs) — FashionMNIST
* [x] Model Saving & Loading
* [ ] Transfer Learning
* [ ] Model Deployment

---

## 🛠️ Tech Stack

Python • PyTorch • NumPy • Matplotlib • Scikit-learn • torchmetrics • mlxtend • Google Colab
