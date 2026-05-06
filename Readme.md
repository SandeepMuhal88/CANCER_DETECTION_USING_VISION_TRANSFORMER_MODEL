# 🔬 Cancer Detection – Chest X-Ray AI

Pneumonia vs Normal classification using **ResNet-18** (Transfer Learning) trained on the
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset.

---

## 📁 Project Structure

```
├── Backend/
│   └── train.py            ← Training script
├── Data/
│   └── chest_xray/chest_xray/
│       ├── train/          ← Training images (NORMAL / PNEUMONIA)
│       ├── val/            ← Validation images
│       └── test/           ← Test images
├── models/
│   └── xray_model.pth      ← Saved model weights (after training)
├── Notebook/
│   └── Notebook_01.ipynb   ← Evaluation, metrics, GradCAM
├── reports/                ← Auto-generated plots
├── app.py                  ← Flask web app
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
cd Backend
python train.py
```
> Trains for 10 epochs, saves best model to `models/xray_model.pth`

### 3. Run the web app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

### 4. Evaluate in Jupyter
```bash
jupyter notebook Notebook/Notebook_01.ipynb
```

---

## 🧠 Model Details

| Property       | Value                          |
|----------------|-------------------------------|
| Architecture   | ResNet-18 (Transfer Learning)  |
| Input Size     | 224 × 224 RGB                  |
| Classes        | NORMAL · PNEUMONIA             |
| Optimizer      | Adam (lr=1e-4)                 |
| Loss           | CrossEntropyLoss               |
| Scheduler      | StepLR (step=5, γ=0.5)        |

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It is not a certified medical
device. Do not use it to make clinical decisions.
