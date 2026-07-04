# Brain Tumor Classification (BraTS) using MobileNetV2 + Grad-CAM

This project implements a lightweight and fast binary classifier for brain MRI images, distinguishing between 'tumor' and 'no-tumor' cases. It leverages the MobileNetV2 architecture with transfer learning and incorporates Grad-CAM for model explainability.

## Project Overview

The goal of this project is to develop an efficient and accurate deep learning model for brain tumor detection from MRI scans. The solution is designed to be runnable end-to-end in Google Colab, demonstrating a complete pipeline from data downloading and preparation to model training, evaluation, and interpretability using Grad-CAM.

### Key Features:
-   **Lightweight Model**: Utilizes MobileNetV2, a highly efficient convolutional neural network, for fast inference.
-   **Transfer Learning**: Employs pre-trained ImageNet weights to boost performance on a relatively smaller dataset.
-   **Automated Data Splitting**: Automatically splits a flat dataset into training, validation, and testing sets (70/15/15).
-   **Data Augmentation**: Applies real-time data augmentation during training to improve model generalization.
-   **Comprehensive Evaluation**: Provides a full suite of classification metrics, confusion matrix, and ROC curve.
-   **Explainable AI (Grad-CAM)**: Visualizes the regions of an image that are most influential in the model's prediction, offering insights into its decision-making process.
-   **Model Persistence**: Saves and loads the trained model for future use or deployment.

## Setup and Installation

This project is designed to run in a Google Colab environment. 

1.  **Clone this repository** (if applicable) or open the notebook in Google Colab.
2.  **Install `kagglehub`**: This library is used to download the dataset directly from Kaggle.

    ```bash
    pip install kagglehub
    ```

3.  **Define `DATASET_DIR`**: A directory to store the downloaded and prepared dataset. For example:

    ```python
    DATASET_DIR = "/content/dataset"
    ```

## Data

The dataset used is `navoneel/brain-mri-images-for-brain-tumor-detection` from Kaggle. The notebook automates the download and preparation process.

### Dataset Structure:

The script expects a flat directory structure like this:

```
/content/dataset/
    no/        -> MRI images WITHOUT a tumor
    yes/       -> MRI images WITH a tumor
```

If your dataset is already split into `train/val/test` folders, set `PRESPLIT = True` in the notebook and adjust the `TRAIN_DIR`, `VAL_DIR`, `TEST_DIR` variables accordingly.

### Data Preprocessing and Augmentation:
-   Images are resized to `224x224` pixels.
-   Pixel values are normalized using `MobileNetV2`'s `preprocess_input` function (scales to `[-1, 1]`).
-   **Training data** undergoes augmentation: horizontal flips, small rotations (`10` degrees), small zooms (`0.10`), and `nearest` fill mode.
-   **Validation and Test data** are only preprocessed.

## Model Architecture

The model is built upon a pre-trained `MobileNetV2` backbone, with a custom classification head.

-   **Backbone**: `MobileNetV2` loaded with `imagenet` weights, excluding the top classification layer.
-   **Fine-tuning**: The last `20` layers of the `MobileNetV2` backbone are unfrozen and fine-tuned during training.
-   **Custom Head**: Consists of `GlobalAveragePooling2D`, `Dropout` (`0.3`), a `Dense` ReLU layer (`64` units), another `Dropout` (`0.2`), and a final `Dense` sigmoid layer (`1` unit) for binary classification.

## Training

The model is compiled with the `Adam` optimizer (learning rate `1e-4`) and `binary_crossentropy` loss, tracking `accuracy` and `AUC` metrics.

-   **Epochs**: `20` epochs.
-   **Batch Size**: `32`.
-   **Callbacks**: 
    -   `EarlyStopping`: Monitors `val_loss` with `patience=4`, restoring the best weights.
    -   `ReduceLROnPlateau`: Reduces learning rate by a factor of `0.5` if `val_loss` does not improve for `2` epochs.
    -   `ModelCheckpoint`: Saves the best model weights based on `val_loss`.

## 🚀 Evaluation Results

After training, the model was evaluated on the test set. Here are the key performance metrics:

```
==================================================
TEST SET EVALUATION
==================================================
Accuracy : 0.6750
Precision: 0.7200
Recall   : 0.7500
F1-score : 0.7347
ROC-AUC  : 0.7422

Confusion Matrix:
[[ 9  7]
 [ 6 18]]

Classification Report:
              precision    recall  f1-score   support

          no       0.60      0.56      0.58        16
         yes       0.72      0.75      0.73        24

    accuracy                           0.68        40
   macro avg       0.66      0.66      0.66        40
weighted avg       0.67      0.68      0.67        40
```

### Interpretation:
-   The model achieved an **accuracy of 67.50%** on the unseen test data. While above random chance, there's room for improvement to reach the target of >= 80%.
-   The **ROC-AUC of 0.7422** indicates a moderately good ability to distinguish between positive and negative classes.
-   The confusion matrix shows `9` correctly classified 'no' cases and `18` correctly classified 'yes' cases.

## Explainable AI: Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize the specific regions in the MRI images that the model focuses on when making a prediction. This helps in understanding the model's decision-making process.

The notebook demonstrates Grad-CAM by displaying the original MRI, the heatmap generated by Grad-CAM, and an overlay of the heatmap on the original image for both 'tumor' and 'no-tumor' examples from the test set.

## Prediction on New Images

Functions are provided to load a single MRI image from disk, preprocess it, and obtain a prediction (class label and confidence score) using the trained model.

## Model Saving and Loading

The trained model is saved in the Keras native format (`.keras`) to `/content/brain_tumor_mobilenetv2_final.keras`. It can be easily loaded back for further inference or deployment without retraining.

```python
# Example of loading the saved model
loaded_model = keras.models.load_model(FINAL_MODEL_PATH)
```

## Usage

Simply run all cells in the provided Colab notebook sequentially. The notebook handles data download, splitting, model definition, training, evaluation, and Grad-CAM visualization automatically.
