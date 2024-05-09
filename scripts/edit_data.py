import numpy as np

data_file = "npy_files/X_B.npy"
data = np.load(data_file, allow_pickle=True).item()


avg_distances = {}
for key, value in data.items():
    _avg_distance = []
    for sub_track in value:
        distance = np.linalg.norm(sub_track[:2] - sub_track[2:4])
        _avg_distance.append(distance)
    avg_distances[key] = np.mean(_avg_distance)

np.save("npy_files/avg_distances.npy", avg_distances)

tracks_with_low_avg_distance = []
for key, value in avg_distances.items():
    if value < 1:
        tracks_with_low_avg_distance.append(key)
print(len(tracks_with_low_avg_distance))

new_data = {}
new_data2 = {}
for key,value in data.items():
    if key in tracks_with_low_avg_distance:
        new_data2[key] = value
    else:
        new_data[key] = value

np.save("npy_files/X_B_filtered.npy", new_data)
np.save("npy_files/X_B_to_short.npy", new_data2)
        