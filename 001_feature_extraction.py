"""
Feature Extraction using openSMILE with eGeMAPS v2.0
Extracts 88 features from audio files in audio_openSMILE/0 and audio_openSMILE/1
Creates a labeled dataset for Random Forest classification
"""

import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import logging
from tqdm import tqdm

# Configuration
AUDIO_FOLDERS = {
    0: './audio_openSMILE/0',
    1: './audio_openSMILE/1'
}
OUTPUT_FILE = './dataset/feature_extraction_dataset.csv'
TEMP_DIR = tempfile.gettempdir()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_opensmile_installation():
    """Check if openSMILE is installed and accessible."""
    try:
        result = subprocess.run(
            ['SMILExtract', '-h'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except FileNotFoundError:
        logger.error("openSMILE is not installed or not in PATH. Please install it with: pip install opensmile")
        return False


def extract_features_from_audio(audio_path, output_csv_path=None):
    """
    Extract eGeMAPS v2.0 features from an audio file using openSMILE.
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file
    output_csv_path : str, optional
        Path to save the features CSV. If None, uses a temporary file.
    
    Returns:
    --------
    dict or None
        Dictionary containing feature names and values, or None if extraction fails
    """
    if output_csv_path is None:
        output_csv_path = os.path.join(TEMP_DIR, f'temp_features_{np.random.randint(0, 10000)}.csv')
    
    try:
        # Run openSMILE with eGeMAPS v2.0 configuration
        # Using risefall-sil configuration for eGeMAPS v2.0 features
        cmd = [
            'SMILExtract',
            '-C', 'config/eGeMAPS_v02_RiseAndFall.conf',
            '-I', audio_path,
            '-O', output_csv_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
            text=True
        )
        
        if result.returncode != 0:
            logger.warning(f"openSMILE extraction failed for {audio_path}: {result.stderr}")
            return None
        
        # Read the extracted features
        df = pd.read_csv(output_csv_path)
        
        # Remove the 'name' column (usually the first column with filename)
        if 'name' in df.columns:
            df = df.drop('name', axis=1)
        
        features_dict = df.iloc[0].to_dict()
        
        # Clean up temporary file
        if os.path.exists(output_csv_path):
            os.remove(output_csv_path)
        
        return features_dict
    
    except subprocess.TimeoutExpired:
        logger.error(f"openSMILE extraction timed out for {audio_path}")
        return None
    except Exception as e:
        logger.error(f"Error extracting features from {audio_path}: {str(e)}")
        return None


def extract_features_faster(audio_path):
    """
    Alternative faster extraction using python-opensmile library.
    Falls back to this if command-line openSMILE extraction fails.
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file
    
    Returns:
    --------
    dict or None
        Dictionary containing feature names and values
    """
    try:
        import opensmile
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        features = smile.process_file(audio_path)
        return features.iloc[0].to_dict()
    except ImportError:
        logger.error("python-opensmile not installed. Please install with: pip install opensmile")
        return None
    except Exception as e:
        logger.error(f"Error in faster extraction for {audio_path}: {str(e)}")
        return None


def build_dataset():
    """
    Build the complete dataset by extracting features from all audio files.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing all extracted features and labels
    """
    dataset_records = []
    
    for label, folder_path in AUDIO_FOLDERS.items():
        if not os.path.exists(folder_path):
            logger.warning(f"Folder not found: {folder_path}")
            continue
        
        # Get all audio files (common formats)
        audio_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.mpeg', '.aac'))
        ])
        
        logger.info(f"Found {len(audio_files)} audio files in {folder_path} (label: {label})")
        
        # Extract features from each audio file
        for audio_file in tqdm(audio_files, desc=f"Processing label {label}"):
            audio_path = os.path.join(folder_path, audio_file)
            logger.info(f"Extracting features from: {audio_path}")
            
            # Try command-line extraction first, fall back to python library
            features = extract_features_from_audio(audio_path)
            if features is None:
                logger.info("Trying faster extraction method...")
                features = extract_features_faster(audio_path)
            
            if features is not None:
                # Add label to the features dictionary
                features['label'] = label
                features['filename'] = audio_file
                dataset_records.append(features)
                logger.info(f"Successfully extracted features. Feature count: {len(features) - 2}")  # -2 for label and filename
            else:
                logger.warning(f"Failed to extract features from {audio_file}")
    
    if not dataset_records:
        logger.error("No features were extracted. Please check your openSMILE installation.")
        return None
    
    # Create DataFrame
    dataset = pd.DataFrame(dataset_records)
    logger.info(f"Dataset shape: {dataset.shape}")
    logger.info(f"Feature columns (excluding label and filename): {len(dataset.columns) - 2}")
    
    return dataset


def save_dataset(dataset, output_path=OUTPUT_FILE):
    """
    Save the dataset to a CSV file.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        The feature dataset to save
    output_path : str
        Path where to save the CSV file
    """
    if dataset is None:
        logger.error("Cannot save None dataset")
        return False
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        dataset.to_csv(output_path, index=False)
        logger.info(f"Dataset saved successfully to: {output_path}")
        logger.info(f"Dataset info:")
        logger.info(f"  - Total samples: {len(dataset)}")
        logger.info(f"  - Total features (including label and filename): {len(dataset.columns)}")
        logger.info(f"  - Class distribution:\n{dataset['label'].value_counts()}")
        return True
    except Exception as e:
        logger.error(f"Error saving dataset: {str(e)}")
        return False


def load_dataset(input_path=OUTPUT_FILE):
    """
    Load the saved dataset from a CSV file.
    
    Parameters:
    -----------
    input_path : str
        Path to the saved CSV file
    
    Returns:
    --------
    pd.DataFrame or None
        The loaded dataset or None if file doesn't exist
    """
    try:
        dataset = pd.read_csv(input_path)
        logger.info(f"Dataset loaded successfully from: {input_path}")
        logger.info(f"Dataset shape: {dataset.shape}")
        return dataset
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {input_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Starting Feature Extraction with openSMILE eGeMAPS v2.0")
    logger.info("=" * 60)
    
    # Check openSMILE installation
    if not check_opensmile_installation():
        logger.warning("Command-line openSMILE not found. Will attempt to use python-opensmile library.")
    
    # Build dataset
    logger.info("\nBuilding dataset...")
    dataset = build_dataset()
    
    if dataset is None:
        logger.error("Failed to build dataset")
        return False
    
    # Save dataset
    logger.info("\nSaving dataset...")
    success = save_dataset(dataset, OUTPUT_FILE)
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("Feature extraction completed successfully!")
        logger.info("=" * 60)
        return True
    else:
        logger.error("Failed to save dataset")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
