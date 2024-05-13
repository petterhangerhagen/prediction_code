import numpy as np
import matplotlib.pyplot as plt
from cubic_spline_interpolation import cubic_spline
import scipy.io as sio

# data_file = "/home/aflaptop/Documents/Radar-data-processing-and-analysis/code/npy_files/valid_tracks.npy"
data_file = "/home/aflaptop/Documents/radar_tracker/Radar-data-processing-and-analysis/code/npy_files/valid_tracks.npy"
data = np.load(data_file, allow_pickle=True)

# print(data.item())
# print(type(data))

# Assume data is a dictionary-like object stored inside a numpy array, and you need the first element
T = []
if isinstance(data.item(), dict):
    tracks_dict = data.item()
    for k, (key, track) in enumerate(tracks_dict.items()):
        print(f"Track {key}")
        track = np.array(track)
        xs = track[:, 0]
        ys = track[:, 1]
        S = cubic_spline(xs, ys)
        T.append(S)
else:
    print("Error: Data does not contain a dictionary")


num_cells = len(T)

# Create a numpy object array to store matrices
cell_array = np.empty((num_cells,), dtype=object)

# Populate the object array with random 382x6 matrices
for i in range(num_cells):
    cell_array[i] = T[i]

# Create a dictionary to hold the cell array with a key corresponding to MATLAB variable name
data = {'X_B2': cell_array}

# Save the dictionary to a .mat file
sio.savemat('X_B2_2.mat', data)

