🧠 PET Image Classification for Lung Cancer Subtypes


📌 Project Overview

This project explores lung cancer subtype classification using PET scan data. The goal is to extract meaningful radiomics-inspired features from medical images and train a machine learning model to distinguish between cancer types.

The pipeline demonstrates:

Medical image preprocessing (DICOM handling)

Feature extraction (intensity, texture, shape proxies)

Machine learning classification

Model evaluation using clinically relevant metrics

Dataset

Source: Lung Cancer PET/CT dataset (DICOM format)

Classes:
Adenocarcinoma
Squamous Cell Carcinoma
Small Cell Carcinoma
Large Cell Carcinoma
Total samples: ~18,500 slices

Important limitation
The dataset does not provide tumor segmentation masks or patient IDs.
Therefore:
Feature extraction is performed on full images and proxy regions
True tumor localization is not possible
Potential data leakage may exist due to slice-level splitting

⚙️ Methodology
1. Data Loading & Preprocessing
DICOM images loaded using pydicom
Pixel intensities normalized to [0, 1]
Empty or invalid images removed

2. Feature Extraction
To represent each scan numerically, three groups of features were extracted:
🟦 Intensity Features
Mean intensity
Standard deviation
Maximum intensity
10th and 90th percentiles
🟩 Texture Features (GLCM)
Contrast
Homogeneity
Energy

These capture spatial intensity patterns in the image.

🟨 Shape Proxy
Approximate “high-intensity region volume” using percentile thresholding

⚠️ This is a proxy feature, not true tumor segmentation.

🔹 3. Model Training
Model: Random Forest Classifier
Parameters:
n_estimators=200
class_weight="balanced"
Validation:
80/20 train-test split
5-fold stratified cross-validation


📈 Results
🔹 Cross-Validation
CV Scores: [0.794, 0.788, 0.789, 0.801, 0.799]
Mean CV: 0.794

🔹 Classification Report
                         precision    recall  f1-score   support
Adenocarcinoma              0.81      0.95      0.87      2000
Large Cell Carcinoma        0.73      0.33      0.46       100
Small Cell Carcinoma        0.82      0.59      0.69       600
Squamous Cell Carcinoma     0.80      0.69      0.74      1000

accuracy                                         0.81      3700
macro avg                  0.79      0.64      0.69
weighted avg               0.81      0.81      0.80



📊 Visualizations
🔹 Confusion Matrix

(Insert plot from notebook here)

🔹 ROC Curves

(Insert ROC curve plot here)

🔹 Feature Importance

(Insert feature importance plot here)

🔹 Sample PET Slices

(Insert sample grayscale images here)

Note: No ROI overlays are shown to avoid misleading interpretations.

🧠 Key Insights
The model achieves ~81% accuracy, showing that basic radiomics features contain predictive signal.
Performance varies significantly across classes:
Strong performance on Adenocarcinoma
Weak performance on Large Cell Carcinoma (likely due to class imbalance)
Texture and intensity features contribute most to model decisions.

⚠️ Limitations
This project intentionally highlights real-world challenges in medical imaging:
❌ No tumor segmentation masks → no true ROI-based features
❌ Slice-level splitting → potential data leakage
❌ Class imbalance affects minority class performance
❌ PET intensity not standardized (no SUV normalization)


🚀 Future Improvements
Use patient-level splitting to avoid leakage
Incorporate true tumor segmentation masks
Apply deep learning (CNNs / 3D models)
Use PyRadiomics for standardized feature extraction
Combine PET with CT (multimodal learning)

💼 Relevance for Medical Imaging Roles
This project demonstrates:
Understanding of medical imaging pipelines
Experience with DICOM data handling
Ability to extract and interpret radiomics features
Proper use of ML evaluation metrics
Awareness of clinical and methodological limitations

✅ Final Note
