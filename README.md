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
| Evaluation Metrics | Accuracy, Precision |
| Computer Vision (CNNs) | Conv2d, MaxPool2d, TinyVGG, FashionMNIST |
| Data Augmentation | V2 Transforms, generalization, preventing overfitting|
| Transfer Learning | Feature extraction, frozen backbones, pre-trained weights (ResNet/EfficientNet) |
| Model Tuning | Hyperparameters (LR, Weight Decay), Schedulers, Label Smoothing, Dropout | 
| Model Persistence | state_dict() save & load, .pth files |

---

## 🖼️ FashionMNIST — Computer Vision

Trained and compared **3 architectures** on FashionMNIST, from a linear baseline to a full CNN.

| Model | Loss | Accuracy | Time |
|---|---|---|---|
| V0 — Baseline Linear | 0.4798 | 83.41% | 28.68s |
| V1 — Linear + ReLU | 0.6850 | 75.02% | 31.95s |
| **V2 — TinyVGG CNN**  | **0.3273** | **88.34%** | 37.24s |

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
#### 🛠️ Architecture Comparison: Custom TinyVGG vs. ResNet-18
This comparison evaluates a **Custom TinyVGG** (trained from scratch) against a **ResNet-18** utilizing Transfer Learning.

* **ResNet-18 (Transfer Learning):** By leveraging pre-trained ImageNet weights and training only the custom classifier head, this model achieved rapid convergence and a superior accuracy of ~65%. The visible spikes in test metrics suggest some sensitivity to specific validation samples, likely due to the frozen backbone's fixed feature set.

* **Custom TinyVGG (Baseline):** A lightweight architecture (20 hidden units, BatchNorm, and Dropout) trained from scratch. Despite using Label Smoothing (0.1) and a StepLR scheduler to improve generalization, it struggled to compete with the pre-trained features, peaking at ~50% accuracy.

# Tuning & Regularization Insights
| FeatureCustom | TinyVGG | ResNet-18 (Transfer)|
|--|--|--|
| Strategy | From Scratch | Frozen Backbone + New Head |
| Regularization | BatchNorm, Dropout (0.5), Weight Decay, Label Smoothing | Pre-trained Weights, Dropout (0.3) |
| Optimization | Adam + StepLR | SchedulerAdam |

<img width="1222" height="855" alt="Untitled" src="https://github.com/user-attachments/assets/8459a602-052a-4273-ae9a-c335c1d183d6" />
  
> **Conclusion:** Transfer Learning via ResNet-18 provided a significantly higher performance ceiling and faster convergence compared to the custom scratch-built architecture.

## 🛠️ Model Tuning
I evaluated two fine-tuning strategies to optimize performance:
* **Frozen EfficientNet:** Acted as a fixed feature extractor, maintaining a stable ~90% accuracy and low loss.

* **Unfreeze Last Block:** Fine-tuned only the final convolutional block. It showed a steady learning curve, climbing from ~40% to ~70% accuracy by epoch 10.

<img width="1601" height="855" alt="Untitled" src="https://github.com/user-attachments/assets/da953236-f546-444c-bdf7-5c56fe899be9" />

> **Observation:** The frozen backbone offers immediate stability, while unfreezing the last block demonstrates a clear trajectory for specialized learning with more epochs.

---

## 📌 Progress Tracker

* [x] Binary Classification
* [x] Multiclass Classification
* [x] Computer Vision (CNNs)
* [x] Custom DataLoader
* [x] Model Saving & Loading
* [x] Transfer Learning
* [x] Model Deployment

---

## 🛠️ Tech Stack

Python • PyTorch • NumPy • Matplotlib • Scikit-learn • torchmetrics • mlxtend • Google Colab
