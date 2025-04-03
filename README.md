(Note : Code for brain tumor and brain anatomy segmentation is present inside EMTENSOR folder)

# NeuroSEG - Brain Tumor Segmentation Tool


## 🧠 [Try the live demo](https://neuroseg.streamlit.app/)

NeuroSEG is an interactive web application that performs automatic segmentation of brain tumors from MRI scans. Using advanced deep learning technology, this tool provides medical professionals and researchers with rapid, accurate tumor segmentation to assist in diagnosis and treatment planning.

## Features

- **Upload your own FLAIR.nii files** or use our sample data
- **Instant tumor segmentation** using a 3D U-Net deep learning model
- **Interactive visualization** with overlay and side-by-side views
- **Multi-view analysis** with axial, coronal, and sagittal planes
- **Tumor statistics** including volume measurements and percentage metrics
- **User-friendly interface** accessible to both clinical and research users

## How It Works

NeuroSEG uses a 3D U-Net architecture trained on the BraTS (Brain Tumor Segmentation) dataset to automatically identify and segment tumor regions in FLAIR MRI sequences. The model distinguishes between:

- Non-enhancing tumor
- Edema
- Enhancing tumor

## Usage Instructions

1. Visit [https://neuroseg.streamlit.app/](https://neuroseg.streamlit.app/)
2. Upload a FLAIR.nii or nii.gz file (limit 200MB) or use the provided sample data
3. Wait for the segmentation to complete
4. Explore the results through the interactive visualization tools
5. Analyze the tumor statistics provided in the dashboard

## Technical Details

- Framework: Streamlit
- Deep Learning: TensorFlow/Keras
- Segmentation Model: 3D U-Net
- Input Format: NIfTI (.nii, .nii.gz)

## Limitations

- Current version is only trained on limited data due to resource constraints
- Processing time depends on server load and file size
- Maximum file size: 200MB

## Future Developments

- Volume measurement tools
- Integration with other imaging modalities
- Longitudinal analysis capabilities
