import numpy as np
import matplotlib.pyplot as plt
from utilities import eucldian_distance

start_points = np.array([0, 0, 1, 1])
data_points_1 = np.array([-1, 0, 0, 1, 1, 2])
data_points_2 = np.array([2, 1, 1, 0, 0, -1])


euclidean_distance = eucldian_distance(start_points, data_points_1[2:6])
print(euclidean_distance)

eucldian_distance = eucldian_distance(start_points, data_points_2[2:6])
print(eucldian_distance)

# '#1f77b4', '#2ca02c'

fig, ax = plt.subplots(figsize=(11, 7.166666))
x = [start_points[0], start_points[2]]
y = [start_points[1], start_points[3]]
ax.scatter(x,y, s=100, marker="o", color="black", label="Current points")
ax.plot(x, y, linestyle='--', color='black')
ax.text(x[0] - 0.05, y[0], r'$\mathbf{\hat{p}}_{k-1}$', fontsize=12, color='black', verticalalignment='bottom', horizontalalignment='right')
ax.text(x[1] - 0.05, y[1], r'$\mathbf{\hat{p}}_{k}$', fontsize=12, color='black', verticalalignment='bottom', horizontalalignment='right')


color = '#1f77b4'
x = [data_points_1[0], data_points_1[2], data_points_1[4]]
y = [data_points_1[1], data_points_1[3], data_points_1[5]]
ax.scatter(x,y, s=100, marker="o", color=color, label="Sub-trajectory 1")
ax.plot(x, y, linestyle='--', color='black')
ax.quiver(data_points_1[2], data_points_1[3], data_points_1[4] - data_points_1[2], data_points_1[5] - data_points_1[3], scale_units='xy', angles='xy', scale=1, color=color)
ax.text(x[0] - 0.05, y[0], r'$\mathbf{p}_{1,n-2}$', fontsize=12, color='black', verticalalignment='bottom', horizontalalignment='right')
ax.text(x[1] - 0.05, y[1], r'$\mathbf{p}_{1,n-1}$', fontsize=12, color='black', verticalalignment='bottom', horizontalalignment='right')
ax.text(x[2] - 0.05, y[2], r'$\mathbf{p}_{1,n}$', fontsize=12, color='black', verticalalignment='bottom', horizontalalignment='right')

color = '#2ca02c'
x = [data_points_2[0], data_points_2[2], data_points_2[4]]
y = [data_points_2[1], data_points_2[3], data_points_2[5]]
ax.scatter(x,y, s=100, marker="o", color=color, label="Sub-trajectory 2")
ax.plot(x, y, linestyle='--', color='black')
ax.quiver(data_points_2[2], data_points_2[3], data_points_2[4] - data_points_2[2], data_points_2[5] - data_points_2[3], scale_units='xy', angles='xy', scale=1, color=color)
ax.text(x[0] - 0.05, y[0], r'$\mathbf{p}_{2,n-2}$', fontsize=12, color='black', verticalalignment='bottom', horizontalalignment='right')
ax.text(x[1] - 0.05, y[1], r'$\mathbf{p}_{2,n-1}$', fontsize=12, color='black', verticalalignment='bottom', horizontalalignment='right')
ax.text(x[2] - 0.05, y[2], r'$\mathbf{p}_{2,n}$', fontsize=12, color='black', verticalalignment='bottom', horizontalalignment='right')

ax.set_xlim(-1.5, 2.5)
ax.set_ylim(-1.5, 2.5)


plt.legend()
plt.savefig("Images/euclidean_distance_example_plot.png", dpi=300)
plt.show()

