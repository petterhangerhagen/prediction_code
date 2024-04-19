# Vessel Movement Prediction from Radar Tracks

## Overview
This project aims to predict vessel movements in the Trondheim fjord based on radar track data using machine learning techniques. The radar data was captured using a frequency modulated continuous wave (FMCW) radar stationed at Fosenkaia, overlooking the Trondheim City Canal. This dataset offers a unique opportunity to analyze a variety of vessel movements, from large ferries to small kayaks, providing a comprehensive view of maritime traffic patterns in this area.

## Dataset
The dataset, named `X_B.npy` and `X_B.mat`, consists of processed radar measurements stored in NumPy and MATLAB formats respectively. These files contain the radar track data that has been filtered and clustered to represent discrete vessel movements within the canal area.

### Structure
- `X_B.npy`: Contains radar track data in a structured NumPy array format.
- `X_B.mat`: Contains the same data as `X_B.npy` but formatted for MATLAB.

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
