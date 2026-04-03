PET Image Analysis Pipeline (Lung Cancer Dataset)

📌 Overview

In this project, I developed an end-to-end PET image analysis pipeline using a public lung cancer dataset. The goal was to simulate a realistic clinical and research environment, where imaging data is processed, validated, and analyzed to extract meaningful quantitative biomarkers.

Starting from raw DICOM images, the pipeline performs preprocessing, quality control, segmentation of high-uptake regions, and quantitative analysis of tumor characteristics. The final outputs include both visualizations and structured data that could support downstream tasks such as statistical analysis or machine learning.

This project is designed to reflect workflows commonly used in oncology research and clinical trials, providing a practical demonstration of how medical imaging data can be translated into measurable insights.



🗂️ Dataset
This project uses a subset of the Lung PET-CT dataset available on Kaggle.
https://www.kaggle.com/datasets/sshhwweettaa/lung-cancer-ct-pet-subset-dicom-format/data

Image-only dataset (no annotations or clinical metadata) of DICOM images from patients diagnosed with lung cancer and includes both PET and CT modalities.

Key characteristics:
~18,500 DICOM images
Four lung cancer types:
Adenocarcinoma (~10,000 images)
Small Cell Carcinoma (~3,000 images)
Squamous Cell Carcinoma (~5,000 images)
Large Cell Carcinoma (~500 images)

Notes:
The dataset is highly imbalanced, which affects model performance.
Images are stored in DICOM format, commonly used in clinical workflows.
No segmentation masks or labels at pixel level are provided.


⚙️ Pipeline Overview
1. Data Ingestion & Handling
Loaded DICOM images using MONAI
Preserved spatial structure and metadata
Applied random sampling for efficient experimentation
2. Preprocessing
Intensity normalization (scaled to [0, 1])
Channel formatting using MONAI transforms
Resampling to a consistent resolution (simulated)
Noise reduction via Gaussian filtering
3. Quality Control (QC)

Automated validation steps to ensure data reliability:
Slice count verification
Intensity distribution checks
Detection of corrupted or low-signal scans

4. Segmentation (ROI Extraction)
Threshold-based segmentation of high-uptake regions
Binary mask generation for lesion areas
Overlay visualization for qualitative validation
5. Quantitative Analysis

Extraction of clinically relevant imaging biomarkers:

Lesion volume (voxel-based → mL approximation)
Mean uptake (SUV-like approximation)
Maximum uptake
Standard deviation (tumor heterogeneity indicator)

6. Visualization
2D slice overlays using Matplotlib
Interactive 3D exploration with Napari
Heatmaps highlighting metabolic activity

7. Data Structuring
Exported results to CSV format
Created a structured dataset suitable for:
Statistical analysis
Machine learning workflows


📊 Example Outputs
Segmentation Overlay

(Insert your saved image here)

![Segmentation](images/segmentation.png)
Quantitative Metrics (per scan)
Lesion volume
Mean uptake
Maximum uptake
🛠️ Technical Stack
Python
MONAI (medical imaging framework)
NumPy
Matplotlib
Napari
🏥 Clinical Relevance

This pipeline reflects key components of PET imaging workflows commonly used in:

Clinical trials
Oncology imaging studies
Drug development and treatment evaluation

Note: Advanced steps such as attenuation correction, scatter correction, and image reconstruction are typically handled by scanner vendor software and are conceptually represented in this project.

🚀 Future Improvements
Deep learning–based segmentation (e.g., MONAI U-Net)
Radiomics feature extraction
Multi-modal PET/CT registration
Automated reporting pipeline

👤 Author
Marcin Szczepański






------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
Overview

This project demonstrates a full pipeline for PET image analysis, including preprocessing, segmentation, quantitative analysis, and visualization. The goal is to simulate a clinical/research workflow similar to those used in PET-based clinical trials and drug development.

Dataset
Public dataset from The Cancer Imaging Archive
Lung cancer PET/CT images (image-only subset)

Classes:
Adenocarcinoma
Small Cell Carcinoma
Large Cell Carcinoma
Squamous Cell Carcinoma

Pipeline Overview
1. Data Ingestion & Handling
Loaded DICOM images using MONAI
Preserved image structure and metadata
Random sampling for efficient experimentation

2. Preprocessing
Intensity normalization (scaled to [0,1])
Channel formatting using MONAI transforms
Resampling to consistent resolution (simulated)
Noise reduction using Gaussian filtering

3. Quality Control (QC)
Automated checks:
Slice count validation
Intensity distribution checks
Basic anomaly detection for corrupted or low-signal scans

4. Segmentation (ROI Extraction)
Threshold-based segmentation of high-uptake regions
Binary mask generation for lesion areas
Overlay visualization for validation

5. Quantitative Analysis
Lesion volume estimation (voxel-based → mL)
Mean and maximum uptake values (SUV-like approximation)
Standard deviation for heterogeneity analysis

6. Visualization
2D slice overlays using Matplotlib
Interactive 3D visualization using Napari
Heatmap overlays highlighting metabolic activity

7. Data Structuring
Results exported to CSV
Structured dataset for downstream analysis or ML workflows
Example Output
PET slice with segmentation overlay

Quantitative metrics per image:
Lesion volume
Mean uptake
Max uptake
Technical Stack
Python
MONAI (medical imaging framework)
NumPy
Matplotlib
Napari
Clinical Context

This pipeline reflects common steps in PET analysis workflows used in:

Clinical trials
Oncology imaging studies
Drug development decision-making

Advanced steps such as attenuation correction, scatter correction, and reconstruction are typically handled by scanner vendor software and are included conceptually in this project.

Future Improvements
Deep learning–based segmentation (MONAI UNet)
Radiomics feature extraction
Multi-modal PET/CT registration
Automated reporting pipeline
