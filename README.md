# Harmonic Radar Fingerprinting for USB/Electronic Devices

## Project Overview
This repository implements **harmonic radar fingerprinting** for USB and electronic devices. The approach leverages electromagnetic emissions (IQ data) to uniquely identify or classify devices based on their unintentional radio-frequency (RF) signatures.

---

## Repository Structure
   File/Folder            | Description                                                                                     |
 |------------------------|-------------------------------------------------------------------------------------------------|
 | **Data/**              | Contains all results from scenario testing. |
 | **GNU Radio/**         | GNU Radio Companion (.grc) files for data acquisition.                     
 | **IQ samples/**        | Raw IQ (In-phase/Quadrature) data samples from the experimental campaign.                            |
 | **ElectromagneticNoise.py** | Script for analyzing or simulating electromagnetic noise from devices.                        |
 | **IQtoSpectrograms.py**     | Converts IQ data into spectrograms for visualization and analysis.                            |
 | **angle.py**           | Handles angle-of-arrival estimation or phase analysis for localization/fingerprinting.         |
 | **splitBalanced.py**   | Splits datasets into balanced subsets (e.g., for training/testing).                            |
 | **splitUnbalanced.py** | Splits datasets without balancing (e.g., for real-world or imbalanced scenarios).              |
 | **trainingAndTesting.py**  | Core script for model training and evaluation.                                                 |
 | **trainingAndTestingOOD.py** | Training/testing with out-of-distribution (OOD) data for robustness evaluation.               |

---

## Key Features
- **Data Collection:** Captures IQ samples from USB/device emissions.
- **Signal Processing:** Uses GNU Radio for real-time or offline processing.
- **Feature Extraction:** Converts IQ data to spectrograms for analysis.
- **Model Training:** Includes scripts for balanced/unbalanced dataset splits and OOD evaluation.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/nathanael-denis/HarmonicRadarClassification.git

2. Install requirements
   ```bash
   pip install torch torchvision matplotlib scikit-learn numpy scipy opencv-python pillow

Workflow

**Data Collection**:

Place IQ samples in IQ samples/ or alternatively, use GNU Radio scripts to capture new data.

- Preprocessing:

Use IQtoSpectrograms.py to generate spectrograms from the IQ data. IQtoSpectrograms will take .iq files in the subdirectory of 
IQ samples and generate spectrograms with the default parameters of the file.

- Data splitting: 

For testing without any data augmentation (Electromagnetic noise or angle), run splitBalanced.py or splitUnbalanced.py.
splitBalanced.py will balance the classes, which is of interest in some scenarios (like testing 16 legitimate USBs against 3 BadUSB with similar data collection)
splitUnbalanced.py will use all the available images regardless of the number per class.
If you apply data augmentation, the script angle.py or ElectromagneticNoise.py will handle splitting; there is no need to apply the Data splitting step.

- Training/Testing:

Run trainingAndTesting.py for standard evaluation.
Use trainingAndTestingOOD.py to test model robustness in the out-of-distribution (OOD) scenario. 
Both scripts will run similarly, but trainingAndTestingOOD.py will also consider the dataset OOD (in addition to test, val, and train)

- Optional - Data augmentation

This part aims to further strengthen the classifier against electromagnetic noise, either from the environment or active obfuscation, and also account for the minimal angle and distance changes.
angle.py will apply an angle transformation framework to emulate changes in angle and distance (both are related in practice). Testing will be performed on augmented data, both with seen data for baseline comparison and on data augmented with angles that the classifier did not see during training (OOD)

   
