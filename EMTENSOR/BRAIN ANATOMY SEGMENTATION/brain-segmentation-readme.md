# Brain Anatomy Segmentation Tool

A Python tool for automated Brain MRI anatomy segmentation using FastSurfer, providing detailed region identification, visualization, and quantitative analysis.

## Overview

This tool provides a streamlined interface for brain MRI segmentation using FastSurfer's deep learning algorithms. It handles the entire segmentation pipeline from dependency installation to visualization and statistical analysis.

Key features:
- Automatic dependency installation and FastSurfer setup
- Multi-view segmentation visualization (axial, coronal, sagittal)
- Comprehensive brain region labeling with color mapping
- Quantitative volume statistics for brain regions
- Enhanced image contrast for better visualization

## Prerequisites

- Python 3.6+
- T1-weighted MRI scan (.nii or .nii.gz format)
- 10GB free disk space
- Internet connection (for downloading dependencies)
- GPU recommended but not required

## Installation

No separate installation is required. The tool automatically handles dependency installation during execution.

## Usage

Run directly from the command line:

```bash
python anatomy_segmentation.py
```

You will be prompted to enter the path to your T1-weighted MRI file.

## Visualization Examples

The tool provides three visualization perspectives:

1. **Original T1 Image**: The raw MRI input
2. **Segmentation Mask**: Color-coded regions
3. **Segmentation Overlay**: Original image with transparent color overlay and region labels

Visualizations are available in three anatomical planes:
- Axial (top-down view)
- Coronal (front-back view)
- Sagittal (side view)

## Brain Region Identification

The segmentation identifies and labels over 100 brain regions, including:
- Cortical regions (frontal, parietal, temporal, occipital)
- Subcortical structures (thalamus, hippocampus, amygdala)
- Ventricles and CSF spaces
- White matter regions
- Cerebellum

## Statistical Output

For each identified region, the tool provides:
- Region name
- Label ID
- Voxel count
- Volume in cubic centimeters (cm³)

## Dependencies

The tool automatically installs:
- yacs (0.1.8)
- torch and torchvision
- plotly
- scikit-image
- nibabel
- matplotlib

## How It Works

1. **Initialization**: Sets up directories and validates input file
2. **Dependency Installation**: Automatically installs required packages
3. **FastSurfer Setup**: Clones and configures the FastSurfer repository
4. **Segmentation**: Processes the T1 MRI using FastSurfer's deep learning models
5. **Verification**: Confirms segmentation completion and file generation
6. **Visualization**: Generates multi-view visualizations with color-coded regions
7. **Statistics**: Calculates volumetric measurements for each brain region
