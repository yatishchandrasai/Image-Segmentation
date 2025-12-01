# Semantic Segmentation on a COCO Subset with MobileNetV2–U-Net (TensorFlow)

This repository contains the code and report for an image segmentation project using a subset of the COCO 2017 dataset.  
The goal is to segment four object classes:

- **person**
- **cat**
- **sports ball**
- **book**

plus **background**, using a **MobileNetV2 encoder + U-Net style decoder** implemented in TensorFlow/Keras.

---

## 1. Project Overview

- **Task:** Multi-class semantic segmentation (pixel-wise classification).
- **Dataset:** COCO 2017 subset (approx. 300 training images, 300 validation images, 30 test images).
- **Labels:** COCO-style `labels.json` with instance polygons per image.
- **Framework:** TensorFlow 2.x, Keras, `pycocotools`, `scikit-image`.
- **Approach:**
  - Convert COCO instance masks into a single semantic mask with labels  
    `0 = background, 1–4 = {person, cat, sports ball, book}`.
  - Train a MobileNetV2–U-Net model to predict these labels for every pixel.

---

## 2. Repository Structure

Example structure (adjust if your repo differs):

```text
.
├── notebooks/
│   └── segmentation_coco_mobilenetv2_unet.ipynb   # Main Colab notebook
├── report/
│   ├── report.pdf                                 # 700–1000 word report
│   └── report_with_code_appendix.docx             # Optional Word version
├── README.md
└── requirements.txt                               # (optional) Python deps
```

> **Note:** The COCO subset (images + JSON) is **not** included here due to size and licensing.  
> You must provide your own copy in Google Drive or locally.

---

## 3. Data Layout

The notebook assumes the following folder structure in Google Drive:

```text
MyDrive/
└── Individual assignment 2/
    ├── train-300/
    │   ├── data/           # training images (.jpg/.png)
    │   └── labels.json     # COCO annotations
    ├── validation-300/
    │   ├── data/           # validation images
    │   └── labels.json
    └── test-30/
        └── *.jpg / *.png   # test images (no labels)
```

You can change the base path in the notebook by editing:

```python
BASE_DATA_PATH = "/content/drive/MyDrive/Individual assignment 2"
```

---

## 4. How to Run (Google Colab)

1. Open the main notebook (e.g. `segmentation_coco_mobilenetv2_unet.ipynb`) in **Google Colab**.
2. Mount Google Drive at the top of the notebook:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```
3. Ensure the dataset is in the expected folder (`Individual assignment 2/` in your Drive).
4. Run the notebook cells in order:
   - Data loading and EDA
   - Building semantic masks from COCO polygons
   - `tf.data` pipeline (generator + batching)
   - Model construction (MobileNetV2 encoder, U-Net decoder)
   - Training with callbacks
   - Evaluation: per-class IoU on validation set
   - Visualisation of predictions on validation and test images

---

## 5. Model Details

- **Encoder:** `tf.keras.applications.MobileNetV2` (`include_top=False`, `weights="imagenet"`).
- **Decoder:** U-Net style upsampling path:
  - 4× `Conv2DTranspose` blocks with BatchNorm + ReLU.
  - Skip connections from MobileNetV2 intermediate layers.
- **Input size:** `256 × 256 × 3`.
- **Output:** Per-pixel logits for **5 classes** (background + 4 objects).
- **Loss:** `SparseCategoricalCrossentropy(from_logits=True)`.
- **Optimiser:** Adam (`learning_rate = 1e-4`).
- **Batch size:** 4.
- **Training tricks:**
  - Encoder frozen initially (to prevent overfitting on small dataset).
  - Early stopping on validation loss.
  - ModelCheckpoint to save best weights.

---

## 6. Results (Example)

Validation Intersection-over-Union (IoU) scores observed:

| Class        | IoU   |
|-------------|-------|
| background  | 0.91  |
| person      | 0.61  |
| cat         | 0.48  |
| sports ball | 0.00  |
| book        | 0.00  |

**Interpretation:**

- The model segments **person** and **cat** reasonably well.
- **sports ball** and **book** perform poorly due to:
  - heavy class imbalance,
  - small object sizes,
  - and using semantic segmentation rather than instance segmentation.

Qualitative visualisations show that the network reliably highlights large people in the scene, producing coarse silhouettes and often ignoring minority-class objects.

---

## 7. Future Work

Potential improvements include:

- Handling class imbalance with:
  - class weighting,
  - focal loss, or
  - class-balanced sampling.
- Stronger augmentation focused on **small objects** (e.g. random crops, scale jitter).
- Fine-tuning the MobileNetV2 encoder rather than keeping it fully frozen.
- Adding **ASPP / atrous convolutions** for better multi-scale context.
- Migrating to a full **instance segmentation** framework (e.g. Mask R-CNN) to take full advantage of COCO’s instance annotations and support per-object masks.

---

## 8. Requirements

Typical dependencies:

- Python 3.8+
- TensorFlow 2.x
- `pycocotools`
- `scikit-image`
- `matplotlib`
- `numpy`

Example installation:

```bash
pip install tensorflow pycocotools scikit-image matplotlib numpy
```

---

## 9. Acknowledgements

This project was developed as part of an individual assignment on **Research Methods and Data Science**, focusing on deep learning–based image segmentation using the COCO dataset.
