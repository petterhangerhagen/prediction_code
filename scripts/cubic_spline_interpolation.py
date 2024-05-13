#imports 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.interpolate import CubicSpline, interp1d 
plt.rcParams['figure.figsize'] =(12,8) 

# Use indices as the parametric variable if no natural 'time' parameter is available
def cubic_spline(x, y):
    t = np.arange(len(x))

    # Set up the spline interpolator for both x and y coordinates
    spline_x = CubicSpline(t, x)
    spline_y = CubicSpline(t, y)

    # Create a fine grid of t values for plotting
    t_fine = np.linspace(t.min(), t.max(), len(x)//2)

    # Evaluate spline
    x_fine = spline_x(t_fine)
    y_fine = spline_y(t_fine)

    S = []
    for i in range(len(x_fine)-2):
        S.append([x_fine[i], y_fine[i],x_fine[i+1], y_fine[i+1], x_fine[i+2], y_fine[i+2]])

    return S
