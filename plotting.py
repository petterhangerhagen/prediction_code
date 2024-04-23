import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

def plot_all_vessel_tracks(X_B):
    """Plot all vessel tracks on the given Axes object.
    
    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        X_B (dict): Dictionary of tracks with each key being a track_id and the value being 
                    numpy arrays of coordinates.
    """

    fig, ax = plt.subplots(figsize=(11, 7.166666))

    # Plotting the occupancy grid'
    data = np.load(f"npy_files/occupancy_grid_without_dilating.npy",allow_pickle='TRUE').item()
    occupancy_grid = data["occupancy_grid"]
    origin_x = data["origin_x"]
    origin_y = data["origin_y"]

    colors = [(1, 1, 1), (0.8, 0.8, 0.8)]  # Black to light gray
    cm = LinearSegmentedColormap.from_list('custom_gray', colors, N=256)
    ax.imshow(occupancy_grid, cmap=cm, interpolation='none', origin='upper', extent=[0, occupancy_grid.shape[1], 0, occupancy_grid.shape[0]])
    
    # Extract points for plotting
    points = [X_B[key][:, 0:2] for key in X_B]
    points = np.concatenate(points, axis=0)  # combine all track points into a single array
    xs = np.array(points[:, 0]) + origin_x  # shift x-coordinates by origin_x
    ys = np.array(points[:, 1]) + origin_y  # shift y-coordinates by origin_y
    scatter = ax.scatter(xs, ys, s=0.01, marker=".", color='#1f77b4')
    
    ax.set_xlim(origin_x-120,origin_x + 120)
    ax.set_ylim(origin_y-140, origin_y + 20)
    ax.set_aspect('equal')
    ax.set_xlabel('East [m]',fontsize=15)
    ax.set_ylabel('North [m]',fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()

    # reformating the x and y axis
    x_axis_list = np.arange(origin_x-120,origin_x+121,20)
    x_axis_list_str = []
    for x in x_axis_list:
        x_axis_list_str.append(str(int(x-origin_x)))
    plt.xticks(x_axis_list, x_axis_list_str)

    y_axis_list = np.arange(origin_y-140,origin_y+21,20)
    y_axis_list_str = []
    for y in y_axis_list:
        y_axis_list_str.append(str(int(y-origin_y)))
    plt.yticks(y_axis_list, y_axis_list_str)

    ax.grid(True)
    
    # Create a legend with a larger marker
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='All tracks',
                            markerfacecolor=scatter.get_facecolor()[0], markersize=10)]
    ax.legend(handles=legend_elements, fontsize=12, loc='upper left')

    # Save plot to file
    save_path = 'Images/plot_all_vessel_tracks.png'
    # plt.savefig(save_path, dpi=400)
    # print(f"Plot saved to {save_path}")

    return ax, origin_x, origin_y, legend_elements

def plot_single_vessel_track(ax, track_id, X_B, origin_x, origin_y):
    """Plot a single vessel track identified by track_id.
    
    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        track_id (int): The ID of the track to plot.
        X_B (dict): Dictionary of tracks.
    """
    track = X_B[track_id]
    x = track[:, 0] + origin_x
    y = track[:, 1] + origin_y
    # points = track[:, 0:2] 

    # Plot the track
    ax.plot(x,y, color='red')
    ax.plot(x[0],y[0], marker='o', color='green')  # Start point
    
    
    # Save plot to file
    save_path = 'Images/plot_single_vessel_track.png'
    # plt.savefig(save_path, dpi=300)
    # print(f"Plot saved to {save_path}")

def plot_predicted_path_new(ax, pred_paths, origin_x, origin_y, legend_elements):
    """Plot the predicted path based on provided points.
    
    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        point_list (list): List of points (x, y) in the predicted path.
    """
    # vertecis = [[-110, -110], [-80, -130], [-30, -115], [0, -120], [0, -90], [40, -60], [60, -50], [95, -20], [80, -10], [40, -8], [-20, -6], [-40, -25], [-52, -58], [-60, -68], [-110, -110]]
    # for vertex in vertecis:
    #     vertex[0] += origin_x
    #     vertex[1] += origin_y
    # for i in range(len(vertecis)-1):
    #     ax.plot([vertecis[i][0], vertecis[i+1][0]], [vertecis[i][1], vertecis[i+1][1]], color='black')

    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Predicted path start',markerfacecolor='red', markersize=10))
    # colors = ['red', 'purple', 'blue', 'green', 'orange', 'black']
    colors = ['#ff7f0e','#1f77b4', '#2ca02c','#c73838','#c738c0',"#33A8FF",'#33FFBD','black']  # Orange, blå, grønn, rød, rosa, lyse blå, turkis

    for i,point_list_element in enumerate(pred_paths):
        point_list = point_list_element.points
        point_array = np.array(point_list)
        xs = point_array[:, 0] + origin_x
        ys = point_array[:, 1] + origin_y
        color_ind = i % len(colors)
        ax.plot(xs, ys, color=colors[color_ind], linewidth=2)
        if i == 0:
            ax.plot(point_array[0, 0] + origin_x, point_array[0, 1] + origin_y, marker='o', color=colors[color_ind], markersize=10)
        legend_elements.append(Line2D([0], [0], color=colors[color_ind], label=f'Predicted path: {point_list_element.probability:.4f}', linewidth=2))
    # point_array = np.array(point_list)
    # xs = point_array[:, 0] + origin_x
    # ys = point_array[:, 1] + origin_y
    # ax.plot(xs, ys, color='red', linewidth=2)
    # ax.plot(point_array[0, 0] + origin_x, point_array[0, 1] + origin_y, marker='o', color='red', markersize=10)  # Start point
    # ax.plot(point_array[0, 0] + origin_x, point_array[0, 1] + origin_y, marker='o', color='red', markersize=10)  # Start point
    ax.legend(handles=legend_elements, fontsize=12, loc='upper left')
    
    # Save plot to file
    save_path = 'Images/plot_predicted_path.png'
    # plt.savefig(save_path, dpi=300)
    # print(f"Plot saved to {save_path}")


def plot_predicted_path(ax, point_list, origin_x, origin_y, legend_elements):
    """Plot the predicted path based on provided points.
    
    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        point_list (list): List of points (x, y) in the predicted path.
    """
    point_array = np.array(point_list)
    xs = point_array[:, 0] + origin_x
    ys = point_array[:, 1] + origin_y
    ax.plot(xs, ys, color='red', linewidth=2)
    ax.plot(point_array[0, 0] + origin_x, point_array[0, 1] + origin_y, marker='o', color='red', markersize=10)  # Start point
    # ax.plot(point_array[0, 0] + origin_x, point_array[0, 1] + origin_y, marker='o', color='red', markersize=10)  # Start point
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Predicted path start',markerfacecolor='red', markersize=10))
    legend_elements.append(Line2D([0], [0], color='red', label='Predicted path', linewidth=2))
    ax.legend(handles=legend_elements, fontsize=12, loc='upper left')
    
    # Save plot to file
    save_path = 'Images/plot_predicted_path.png'
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
