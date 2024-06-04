# Vessel Movement Prediction from Radar Tracks

## Overview
This code is used in my master thesis. It is a simplified version of NCDM, developed by Bjørnar Dalsnes. It is based on his MATLAB code.


## Dataset
The code has two main dataset `X_B_train.npy` and `X_B_test.npy`. `X_B_train.npy` consist of all the tracks established by the multi-target tracker, except the 10 tracks in the test set. `X_B_test.npy` consist 10 handpicked tracks. 

To create these dataset the following scripts can be used:
1. scripts/tracks_to_mat.py - This script uses this: scripts/cubic_spline_interpolation.py
2. scripts/mat_to_npy.py - Converts back into a npy-file
3. scripts/split_train_test.py - Split into 10 tracks in test set and the rest in the training set.

### Key Features
- **Multi-target scenarios**: Data includes scenarios where multiple targets are present within close proximity.
- **Merged measurements**: Instances where signals from separate targets are combined into a single detection.
- **Multipath effects**: Scenarios demonstrating errors caused by signal reflections off the water surface.

## Code Files
This repository includes several scripts used to process and analyze the dataset:
- `data_preprocessing.py`: Preprocesses raw radar data to filter noise and separate distinct target tracks.
- `track_extraction.py`: Applies clustering algorithms to identify and extract individual vessel movements from processed radar data.
- `movement_prediction.py`: Utilizes machine learning models to predict future vessel positions based on historical track data.

## Installation
To run the scripts, you need Python 3.8 or later with the following packages installed:
- NumPy
- Matplotlib
- Scikit-learn
- SciPy

You can install all required packages by running:
```bash
pip install numpy matplotlib scikit-learn scipy
