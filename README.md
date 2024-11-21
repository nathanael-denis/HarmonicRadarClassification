# Harmonic Signal Fingerprinting with Deep Learning

This repository contains the tools and scripts developed for the **Harmonic Signal Fingerprinting** project, which explores the use of harmonic radar and deep learning techniques to classify signals from devices and detect anomalies like hardware Trojans.

## Overview

The project combines signal processing and machine learning to create a robust system for analyzing harmonic signals, transforming raw IQ data into informative images, and training deep learning models for classification tasks. This repository is organized to support the full pipeline, from data preparation to distributed training.

## Repository Structure

### Data
The results of the experimental campaign for the different scenarios considered.

### GNU Radio
Contains GNU Radio workflow for emitting (TX), then capturing (RX) and processing harmonic radar signals. The workflow is used to interface with hardware (HackRF and USRP) and capture IQ data in real-time.

### distributedTrainingUSB.py
This script is designed for **distributed training** of deep learning models using PyTorch. It supports:
- Multi-GPU training with distributed data parallelism (distributedTrainingUSB.py).
- Centralized training for testing (workstationTrainingUSB.py)
- ResNet architectures for USB signal classification.
- Metrics calculation : precision, recall, F1-score, and confusion matrix generation.
  
**Usage:**
```bash
python distributedTrainingUSB.py
