import numpy as np
import scipy.io

# Load AIS data from a .mat file
data = scipy.io.loadmat('X_B2.mat')
# print(data.items())
print(data.keys())
print(data['X_B2'].shape)
track_number = 0
print(data['X_B2'][0, track_number].shape)
# print(data['X_B2'][0, track_number])

X_B = {}

for i in range(data['X_B2'].shape[1]):
    X_B[i] = np.array(data['X_B2'][0, i])

print(X_B[0].shape)
print(len(X_B))

# Save to a .npy file
np.save('X_B.npy', X_B)
