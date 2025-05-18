---

# 🧠 Computer Vision Magic Card Classifier — Project Plan

---

## ✅ Phase 1: Front of Card — Proof of Concept

### 🎴 Target Cards (randomly selected):

* **3 of Diamonds** (`3_d`)
* **King of Spades** (`k_s`)
* **7 of Hearts** (`7_h`)
* **Ace of Clubs** (`a_c`)

### 📸 1. Collect Images of Card Fronts

**Goal**: Get 20–30 varied images per card to simulate real-world conditions.

**Steps**:

* Use `collect_card_images.py` to capture labeled images.
* For each card, collect:

  * [ ] 5 images flat on desk, good lighting
  * [ ] 5 images flat on desk, rotated
  * [ ] 5 hand-held near camera
  * [ ] 5 hand-held further away
  * [ ] 5 under poor lighting or with shadows

**Estimated Time**: \~30 minutes (5–10 minutes per card)

---

### 🧠 2. Train Custom CNN on 4 Cards

**Goal**: Verify classification pipeline works with a small, clean dataset.

**Steps**:

* [ ] Point training script to new image directory (e.g., `data/front_images`)
* [ ] Modify label parsing if needed (e.g., `k_s_20250517_...jpg`)
* [ ] Update number of classes to 4
* [ ] Train for \~10–20 epochs, monitor accuracy

**Estimated Time**: \~15 minutes (plus training time)

---

## 🔁 Phase 2: Back of Cards — Secret Marking Detection

### 🃏 1. Collect Images of Trick Card Backs

**Goal**: Repeat same data collection process with backs of the 4 magic cards.

**Steps**:

* Use `collect_card_images.py` with same labels
* Capture \~20–30 images per back (as above)
* Pay close attention to:

  * Marking visibility
  * Background consistency
  * Camera focus

**Estimated Time**: \~30 minutes

---

### 🧠 2. Train CNN on Trick Card Backs

**Goal**: See if your model can detect hidden symbols in the design.

**Steps**:

* [ ] Use same training pipeline as Phase 1
* [ ] Train and evaluate model
* [ ] (Optional) Use Grad-CAM to see if the model attends to the correct region

**Estimated Time**: \~15–30 minutes

---

## 🎥 Phase 3: Real-Time Detection via YOLO + Classifier

### 📦 1. Add YOLO for Card Detection (Front of Card)

**Goal**: Detect cards in the live webcam feed using YOLO, then classify cropped cards.

**Steps**:

* [ ] Use pretrained YOLOv8 or YOLOv5
* [ ] Fine-tune on images of full cards if needed (optional)
* [ ] Detect bounding boxes for visible cards
* [ ] Crop & resize ROI → feed to trained classifier

**Estimated Time**: 1–2 hours to set up + test

---

### 🧠 2. Classify Front of Card in Real Time

**Goal**: Full pipeline: webcam → detect card → predict card type.

**Steps**:

* [ ] Build loop using OpenCV to read webcam
* [ ] Use YOLO to detect card(s)
* [ ] Classify each ROI using trained model
* [ ] Overlay prediction on live video

**Estimated Time**: \~1–2 hours for basic prototype

---

### 🃏 3. Repeat for Card Backs

**Goal**: Test if real-time detection & classification works using back of cards.

**Steps**:

* [ ] Use YOLO + CNN pipeline again
* [ ] Check if markings are visible enough for detection + classification
* [ ] Evaluate robustness to angle, lighting

**Estimated Time**: 1–2 hours

---

## ⏱️ Summary Timeline

| Task                             | Time Estimate   |
| -------------------------------- | --------------- |
| Image collection (front + back)  | \~1 hour        |
| Initial CNN training/testing     | \~30–45 minutes |
| Real-time detection + classifier | \~3–4 hours     |
| **Total Project Time**           | \~5–6 hours     |

---

Let me know if you want this exported to a `.md` file or turned into a task-tracking checklist in code or Notion format. Ready to dive into the next phase when you are!
