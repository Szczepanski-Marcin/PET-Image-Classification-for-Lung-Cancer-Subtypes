# PET/CT Lung Cancer Classification Pipeline

## Overview
This project implements a complete pipeline for medical image analysis and classification using PET/CT DICOM images. The goal is to extract meaningful features from imaging data and classify different types of lung cancer using a machine learning model.

The pipeline includes:

 - DICOM image processing
 - Image normalization
 - ROI (Region of Interest) extraction
 - Feature engineering
 - Classification using Random Forest
 - Performance evaluation

## Dataset
Subset of PET/CT lung cancer images in DICOM format

Contains four cancer types:

 - Adenocarcinoma
 - Small Cell Carcinoma
 - Squamous Cell Carcinoma
 - Large Cell Carcinoma

**Note:** The dataset is imbalanced, with fewer samples for some classes.

## Pipeline

### 1. Data Loading
 - DICOM files are read using pydicom
 - Images are processed one-by-one for memory efficiency

### 2. Preprocessing
 - Images normalized to range [0, 1]
 - Empty or invalid images removed

### 3. ROI Extraction
 - An adaptive threshold is used for ROI detection:
   ```python
   threshold = np.mean(img) + np.std(img)
   roi = img > threshold
   ```
This adapts to different intensity distributions across images.

### 4. Feature Engineering
For each image, the following features are extracted:

 - ROI Size – number of pixels in the region of interest
 - Mean Intensity – average pixel intensity
 - Max Intensity – maximum pixel value
 - Standard Deviation – intensity variability
 - ROI Mean Intensity – mean intensity within ROI

These features provide a simple approximation of radiomic characteristics.

### 5. Machine Learning Model
 - Random Forest Classifier used for classification
 - Handles non-linear relationships
 - Works well with small feature sets
 - `class_weight="balanced"` to address class imbalance

Dataset split:

 - 80% training
 - 20% testing (stratified)

### 6. Evaluation
 - Performance metrics
 - Precision
 - Recall
 - F1-score
 - Accuracy
 - Confusion Matrix

## Results
| Metric         | Value |
| -------------- | ----- |
| Accuracy       | ~70%  |
| Macro F1-score | ~0.53 |

Key observations:

 - Strong performance on Adenocarcinoma (largest class)
 - Lower performance on Large Cell Carcinoma due to class imbalance
 - Feature engineering improved results (~61% → ~70%)

### Example Visualizations

#### ROI Size Distribution:
```python
plt.hist(df["ROI_Size"], bins=30)
plt.title("ROI Size Distribution")
plt.xlabel("ROI Size")
plt.ylabel("Frequency")
plt.show()
```

#### Confusion Matrix: 
Highlights class imbalance and misclassification patterns

## Key Learnings
 - Medical imaging datasets are often imbalanced
 - Simple features can provide a reasonable baseline performance
 - ROI definition significantly impacts results
 - Memory-efficient processing is essential for large DICOM datasets
 - Iterative improvement (features + preprocessing) leads to measurable gains

## Future Improvements

 - Advanced radiomics features (texture, shape descriptors)
 - Deep learning models (CNNs)
 - Improved ROI segmentation (adaptive thresholding, clustering)
 - Cross-validation and hyperparameter tuning
 - Integration of clinical metadata

## Technologies Used

 - Python
 - NumPy
 - Pandas
 - Matplotlib
 - scikit-learn
 - pydicom

## Project Structure
lung-pet-project/
│
├── data/                  # DICOM dataset (not included)
├── scripts/
│   └── project.py         # Main pipeline
├── README.md

## Author
Biomedical Engineering graduate with experience in:

 - Medical image processing
 - Signal analysis
 - Data analytics

## Notes
This project is intended as a learning and demonstration project for applying data science and machine learning techniques to medical imaging data.