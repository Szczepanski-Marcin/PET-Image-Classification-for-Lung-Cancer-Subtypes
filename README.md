PET Image Analysis Pipeline (Lung Cancer Dataset)
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
