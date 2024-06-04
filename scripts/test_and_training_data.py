import numpy as np

X_B = np.load("/home/aflaptop/Documents/Scripts/prediction_code/npy_files/X_B_valid_tracks.npy", allow_pickle=True).item()

test_tracks = [1053, 1023, 1009, 1007, 947, 969, 945, 931, 756, 755]

X_B_test = {}
X_B_train = {}

for key, value in X_B.items():
    if key in test_tracks:
        X_B_test[key] = value
    else:
        X_B_train[key] = value

np.save("/home/aflaptop/Documents/Scripts/prediction_code/npy_files/X_B_test.npy", X_B_test)
np.save("/home/aflaptop/Documents/Scripts/prediction_code/npy_files/X_B_train.npy", X_B_train)