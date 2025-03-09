# T1w_MRI_preprocessing
 A fast and efficient pipeline designed for preprocessing T1w MRI images


# MRI Preprocessing and Augmentation Pipeline

This repository contains scripts for a standardized MRI preprocessing and data augmentation pipeline. The pipeline ensures consistency across datasets by applying several preprocessing steps to the MRI scans and balances the training dataset through augmentation. The jupyter notebook "example.ipynb" demonstrates the use of these functions. 

There is a non-BIDS compatible pipeline in the utils.py as well but may require minor adjustments based on how your target dataset is stored. 

Note that the function that converts dicom files to NIFTI ones, requires **dcm2niix** to be present in the path directory. 

## Preprocessing Pipeline Overview

1. **N4 Bias Field Correction**  
   - Corrects for intensity inhomogeneity using [SimpleITK](https://simpleitk.org/) (v2.4.0).

2. **Registration to MNI152 Atlas**  
   - Uses [Advanced Normalization Tools (ANTs)](https://github.com/ANTsX/ANTs) (v0.5.3) with the SyN non-linear registration method.  
   - Ensures anatomical alignment across subjects.

3. **Brain Extraction and Skull Stripping**  
   - Performed using [ANTsPyNet](https://github.com/ANTsX/ANTsPyNet).

4. **Resizing**  
   - All images are resized to 160 × 192 × 160 voxels using ANTs.

5. **Intensity Normalization**  
   - Standardizes image intensities across subjects using SimpleITK.

Implementing this standardized preprocessing pipeline is essential to minimize variations introduced by different MRI scanners and acquisition protocols.

## Data Augmentation

To improve model robustness and balance the training dataset, data augmentation is applied with the following techniques:

- **Random Rotations:**  
  Up to ±10 degrees.

- **Spatial Shifts:**  
  Up to 5 voxels along each axis.



