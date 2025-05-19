# 🎴 cardnet — Real-Time Playing Card Classifier

This project builds and deploys a real-time computer vision system to detect and classify playing cards from either their front or back sides — including trick card backs with hidden markings.

---

## 🔧 Features

- Train a CNN or fine-tune ResNet18 on card images
- Visualize predictions and Grad-CAM overlays
- Real-time webcam detection + classification loop
- Easily extendable to new card sets or marks

---

## 🗂 Structure

- `scripts/` — training, data collection, webcam inference
- `cardnet/` — core logic and model code
- `models/` — trained model weights
- `data/` — images (not tracked in Git, but expected here)

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/cardnet
cd cardnet
pip install -r requirements.txt
