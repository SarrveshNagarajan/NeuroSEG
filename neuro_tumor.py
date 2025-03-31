import streamlit as st
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import io
import base64
from PIL import Image
import requests
from pathlib import Path
import matplotlib.cm as cm

# Set page title and layout
st.set_page_config(
    page_title="NeuroSEG - Brain Tumor Segmentation Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved aesthetics
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    h1, h2, h3 {color: #2c3e50;}
    .stButton>button {
        color: #fff;
        background-color: #3498db;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {background-color: #2980b9;}
    .sidebar .sidebar-content {background-color: #f8f9fa;}
    .css-18e3th9 {padding-top: 0;}
    .css-1d391kg {padding-top: 3.5rem;}
    .info-box {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #4caf50;
    }
    .error-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Header with logo/icon
col1, col2 = st.columns([1, 20])
with col1:
    # Placeholder for a brain icon - in a real app, you'd use an actual logo
    st.markdown("", unsafe_allow_html=True)
with col2:
    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 2.5rem; color: white; font-weight: 600;">NeuroSEG - Brain Tumor Segmentation Tool</h1>
        </div>
        """,
        unsafe_allow_html=True
    )





# Functions from the original code
def standardize(image):
    standardized_image = np.zeros(image.shape)
    for z in range(image.shape[2]):
        image_slice = image[:,:,z]
        centered = image_slice - np.mean(image_slice)
        if np.std(centered) != 0:
            centered = centered / np.std(centered)
        standardized_image[:, :, z] = centered
    return standardized_image

def preprocess_image(data):
    # Consistent preprocessing steps
    reshaped_data = data[56:184, 80:208, 13:141, :]
    reshaped_data = reshaped_data.reshape(1, 128, 128, 128, 4)
    return reshaped_data

def load_single_nii_file(file_path):
    """Load and process a single NII file"""
    # Load the file
    img = nib.load(file_path)
    image_data = img.get_fdata()
    image_data = np.asarray(image_data)

    image_data = standardize(image_data)

    data = np.zeros((240, 240, 155, 4))
    
    data[:, :, :, 0] = image_data
    
    return data, img.affine

def dice_coef(y_true, y_pred, epsilon=0.00001):
    from tensorflow.keras import backend as K
    axis = (0, 1, 2, 3)
    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true*y_true, axis=axis) + K.sum(y_pred*y_pred, axis=axis) + epsilon
    return K.mean((dice_numerator)/(dice_denominator))

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def create_class_colormap():
    """Create a custom colormap for tumor classes"""
    # Class 0: Background (transparent)
    # Class 1: Necrotic and non-enhancing tumor core (red)
    # Class 2: Peritumoral edema (green)
    # Class 3: Enhancing tumor (blue)
    colors = [(0, 0, 0, 0), (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
    return cm.colors.ListedColormap(colors)

def visualize_segmentation(original_data, prediction, file_name, selected_slice=None):
    """Create both visualization styles: overlay and non-overlaying for Streamlit"""
    # Use the middle slice by default if none is specified
    if selected_slice is None:
        z_max = original_data.shape[2]-1
        selected_slice = z_max // 2
    
    # Create a colormap for segmentation
    tumor_cmap = create_class_colormap()
    
    # Calculate tumor statistics
    tumor_voxels = np.sum(prediction > 0)
    total_voxels = prediction.size
    tumor_percentage = (tumor_voxels / total_voxels) * 100
    
    # Display tumor statistics
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        st.metric("Tumor Voxels", f"{tumor_voxels:,}")
    with stats_col2:
        st.metric("Total Brain Voxels", f"{total_voxels:,}")
    with stats_col3:
        st.metric("Tumor Percentage", f"{tumor_percentage:.2f}%")
    
    # Create tabs for different visualization styles
    viz_tab1, viz_tab2= st.tabs(["Overlay View", "Side-by-Side View"])
    
    with viz_tab1:
        # Style 1: Overlay visualization
        fig1 = plt.figure(figsize=(12, 5))
        
        # Original image with prediction overlay
        plt.subplot(1, 2, 1)
        plt.imshow(original_data[:, :, selected_slice, 0], cmap='gray')
        plt.title(f"FLAIR with Prediction Overlay - {file_name}")
        mask = prediction[:, :, selected_slice]
        plt.imshow(mask, alpha=0.5, cmap='jet')
        plt.axis('off')
        
        # Segmentation only
        plt.subplot(1, 2, 2)
        im = plt.imshow(prediction[:, :, selected_slice], cmap='jet')
        plt.title("Segmentation Only")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig1)
    
    with viz_tab2:
        # Style 2: Side-by-Side visualization with three views (axial, coronal, sagittal)
        fig2 = plt.figure(figsize=(15, 5))
        
        # Axial view (original)
        plt.subplot(131)
        plt.imshow(original_data[:, :, selected_slice, 0], cmap='gray')
        plt.title(f"Axial View - FLAIR")
        plt.axis('off')
        
        # Coronal view
        plt.subplot(132)
        plt.imshow(original_data[:, selected_slice, :, 0].T, cmap='gray')
        plt.title("Coronal View - FLAIR")
        plt.axis('off')
        
        # Sagittal view
        plt.subplot(133)
        plt.imshow(original_data[selected_slice, :, :, 0].T, cmap='gray')
        plt.title("Sagittal View - FLAIR")
        plt.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Segmentation views
        fig3 = plt.figure(figsize=(15, 5))
        
        # Axial segmentation
        plt.subplot(131)
        plt.imshow(prediction[:, :, selected_slice], cmap='viridis')
        plt.title("Axial - Segmentation")
        plt.axis('off')
        
        # Try to create coronal and sagittal views if dimensions allow
        try:
            # Coronal segmentation
            plt.subplot(132)
            plt.imshow(prediction[:, selected_slice, :].T, cmap='viridis')
            plt.title("Coronal - Segmentation")
            plt.axis('off')
            
            # Sagittal segmentation
            plt.subplot(133)
            plt.imshow(prediction[selected_slice, :, :].T, cmap='viridis')
            plt.title("Sagittal - Segmentation")
            plt.axis('off')
        except:
            st.info("Note: Some views may not be available due to dimensional constraints.")
        
        plt.tight_layout()
        st.pyplot(fig3)
    
    

def process_nii_file(nii_file, model):
    """Process the uploaded NII file and return the segmentation"""
    try:
        # Create a temporary file to save the uploaded file
        temp_path = "temp_upload.nii"
        with open(temp_path, "wb") as f:
            f.write(nii_file.getvalue())
        
        # Load the file
        data, affine = load_single_nii_file(temp_path)
        
        # Preprocess data
        reshaped_data = preprocess_image(data)
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Predict with progress updates
        with st.spinner("Running segmentation model..."):
            status_text.text("Initializing segmentation model...")
            progress_bar.progress(10)
            
            status_text.text("Preprocessing complete, running segmentation...")
            progress_bar.progress(30)
            
            prediction = model.predict(reshaped_data)
            progress_bar.progress(80)
            
            status_text.text("Post-processing segmentation results...")
            prediction_classes = np.argmax(prediction[0], axis=-1)
            progress_bar.progress(100)
            status_text.text("Segmentation complete!")
        
        # Return results
        return reshaped_data[0], prediction_classes
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def get_sample_nii_file():
    """Use the existing sample FLAIR NII file"""
    # Define the path to the sample file - using the specific path you provided
    sample_file_path = Path("sample data/Brats17_2013_31_1_flair.nii")
    
    # Check if the file exists
    if sample_file_path.exists():
        st.success("Using existing sample file.")
        return sample_file_path
    else:
        st.error("Sample file not found. Please check the file path.")
        return None

def load_model():
    """Load the segmentation model"""
    # Function to check if model file exists or needs downloading
    model_path = Path("model/3d_model (1).h5")
    
    # Add a status indicator
    status_container = st.empty()
    status_container.info("Initializing segmentation model...")
    
    
    # Load the model with custom objects
    model = keras.models.load_model(
        model_path,
        custom_objects={
            'dice_coef_loss': dice_coef_loss,
            'dice_coef': dice_coef
        }
    )
    
    status_container.success("Segmentation model loaded successfully!")
    return model

def main():
    # Sidebar
    st.sidebar.title("NeuroSEG")
    
    st.sidebar.markdown("### Instructions")
    st.sidebar.markdown("""
    1. Upload a FLAIR NII file or use our sample data
    2. Wait for the segmentation to complete
    3. View the results and analyze the tumor segmentation
    """)
    
    # Create an expandable section for more details
    with st.sidebar.expander("About the Model"):
        st.markdown("""
        This tool uses a 3D U-Net model trained on the BraTS (Brain Tumor Segmentation) dataset to automatically segment brain tumors from MRI FLAIR sequences.
        The model achieves a Dice score of approximately 0.85 on the validation dataset.
        """)
    
    
    # Add contact information
    st.sidebar.markdown("---")

    
    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # File upload section - with improved styling
    st.header("🔬 Analysis Input")
    
    # Create two tabs: Upload and Sample with improved styling
    tab1, tab2 = st.tabs(["📤 Upload Your File", "📋 Use Sample Data"])
    
    with tab1:
        
        
        
            
        uploaded_file = st.file_uploader("Choose a FLAIR NII file", type=['nii', 'nii.gz'], 
                                            help="Upload a FLAIR MRI sequence in NIfTI format")
        
        
            
        
        if uploaded_file is not None:
            # Process the uploaded file with better feedback
            st.markdown(f"""
            <div class="success-box" style="color: black;">
                <h3>✅ File Uploaded Successfully</h3>
                <p>Filename: {uploaded_file.name}<br>
                File size: {uploaded_file.size/1024:.1f} KB</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add a "Start Analysis" button
            if st.button("🔬 Start Tumor Segmentation Analysis"):
                # Process the file
                original_data, segmentation = process_nii_file(uploaded_file, model)
                
                if original_data is not None and segmentation is not None:
                    # Display segmentation results
                    st.markdown("""
                    <h2 style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; color: black;">
                        📊 Segmentation Results
                    </h2>
                    """, unsafe_allow_html=True)
                    
                    visualize_segmentation(original_data, segmentation, uploaded_file.name)
    
    with tab2:
        
        
        # Create a more attractive button
        sample_col1, sample_col2, sample_col3 = st.columns([1, 2, 1])
        with sample_col2:
            sample_button = st.button("🧬 Load Sample Brain MRI Data", use_container_width=True)
            
        if sample_button:
            with st.spinner("Loading sample file and processing..."):
                # In a real app, download or access a pre-packaged sample file
                sample_file_path = get_sample_nii_file()
                
                if sample_file_path and sample_file_path.exists():
                    # Open the file and process it
                    with open(sample_file_path, "rb") as f:
                        # Create a BytesIO object from the file content
                        file_bytes = io.BytesIO(f.read())
                        file_bytes.name = "sample_flair.nii"
                        
                        # Display sample information
                        st.markdown("""
                        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin-bottom: 20px; color: black;">
                            <h3>Sample Data Information</h3>
                            <p><b>Dataset:</b> BraTS 2017<br>
                            <b>Patient ID:</b> Brats17_2013_31_1<br>
                            <b>Sequence:</b> FLAIR</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Process the file
                        original_data, segmentation = process_nii_file(file_bytes, model)
                        
                        if original_data is not None and segmentation is not None:
                            # Display segmentation results
                            st.markdown("""
                            <h2 style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; color: black;">
                                📊 Segmentation Results
                            </h2>
                            """, unsafe_allow_html=True)
                            
                            visualize_segmentation(original_data, segmentation, "BraTS17_Sample.nii")
                else:
                    st.markdown("""
                    <div class="error-box">
                        <h3>⚠️ Sample File Not Found</h3>
                        <p>The sample file could not be located at the specified path. Please try uploading your own file instead.</p>
                    </div>
                    """, unsafe_allow_html=True)
    


if __name__ == "__main__":
    main()