# Vessel Movement Prediction from Radar Tracks

## Overview
This code is used in my master thesis. It is a simplified version of NCDM, developed by Bjørnar Dalsnes. It is based on his MATLAB code.


## Dataset
The code has two main dataset `X_B_train.npy` and `X_B_test.npy`. `X_B_train.npy` consist of all the tracks established by the multi-target tracker, except the 10 tracks in the test set. `X_B_test.npy` consist 10 handpicked tracks. 

To create these dataset the following scripts can be used:
1. **scripts/tracks_to_mat.py** - This script uses this: scripts/cubic_spline_interpolation.py
2. **scripts/mat_to_npy.py** - Converts back into a npy-file
3. **scripts/split_train_test.py** - Split into 10 tracks in test set and the rest in the training set.

## Code files
The main script is `NCDM.py`, which depends on 
- `utilities.py`
- `check_start_and_stop.py`
- `GMM_components.py`.

`plotting.py` is used to create plots. This scripts is a mess. Recommend create own code for this. 


## Installation
To run the code, you need Python 3.8 or later with the following packages installed:
- NumPy
- Matplotlib
- Scikit-learn
There are maybe some more packages which are needed, but just install the ones needed depending on the error codes. 
