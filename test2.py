
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# Sub-trajectories represented as points in 6D space
# sub_trajectory_1 = [45.74406433, -25.3789196, 41.87245566, -26.47365231, 39.10538641, -28.97222296]
# sub_trajectory_2 = [-80.19976807, -125.30456543, -81.4817506, -124.61915363, -82.02491043, -124.55735761]

sub_trajectory_1 = [45, 45, 55, 55]
# sub_trajectory_2 = [60, 60, 50, 50, 40, 40]
# sub_trajectory_2 = [65, 65, 55, 55, 45, 45]
sub_trajectory_2 = [35, 35, 45, 45]
sub_trajectory_2 = [45, 35, 55, 45]
sub_trajectory_2 = [55, 45, 45, 35]



# Calculating the Euclidean distance between the two sub-trajectories
distance = euclidean(sub_trajectory_1, sub_trajectory_2)
print(f"Euclidean distance between the two sub-trajectories: {distance:.2f}")


fig, ax = plt.subplots(figsize=(11, 7.166666))
ax.plot(sub_trajectory_1[0::2], sub_trajectory_1[1::2], color='red', label='Sub-trajectory 1')
ax.scatter(sub_trajectory_1[0], sub_trajectory_1[1], color='red', label='Start point 1')
ax.plot(sub_trajectory_2[0::2], sub_trajectory_2[1::2], color='blue', label='Sub-trajectory 2')
ax.scatter(sub_trajectory_2[0], sub_trajectory_2[1], color='blue', label='Start point 2')
ax.legend()
plt.show()

