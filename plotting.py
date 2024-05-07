import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import numpy as np
from utilities import make_new_directory
import os
import datetime
from scipy.stats import norm

#  colors = ['#ff7f0e','#1f77b4', '#2ca02c','#c73838','#c738c0',"#33A8FF",'#33FFBD']  # Orange, blå, grønn, rød, rosa, lyse blå, turkis

def start_plot():
    fig, ax = plt.subplots(figsize=(11, 7.166666))

    # Plotting the occupancy grid'
    data = np.load(f"npy_files/occupancy_grid_without_dilating.npy",allow_pickle='TRUE').item()
    occupancy_grid = data["occupancy_grid"]
    origin_x = data["origin_x"]
    origin_y = data["origin_y"]

    colors = [(1, 1, 1), (0.8, 0.8, 0.8)]  # Black to light gray
    cm = LinearSegmentedColormap.from_list('custom_gray', colors, N=256)
    ax.imshow(occupancy_grid, cmap=cm, interpolation='none', origin='upper', extent=[0, occupancy_grid.shape[1], 0, occupancy_grid.shape[0]])
    
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

    return ax, origin_x, origin_y

def plot_all_vessel_tracks(ax, X_B, origin_x, origin_y, save_plot=False):

    # Extract points for plotting
    points = [X_B[key][:, 0:2] for key in X_B]
    points = np.concatenate(points, axis=0)  # combine all track points into a single array
    xs = np.array(points[:, 0]) + origin_x  # shift x-coordinates by origin_x
    ys = np.array(points[:, 1]) + origin_y  # shift y-coordinates by origin_y
    scatter = ax.scatter(xs, ys, s=0.01, marker=".", color='#1f77b4')

    # Create a legend with a larger marker
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='All tracks',
                            markerfacecolor=scatter.get_facecolor()[0], markersize=10)]
    ax.legend(handles=legend_elements, fontsize=12, loc='upper left')

    # Save plot to file
    if save_plot:
        save_path = 'Images/plot_all_vessel_tracks.png'
        plt.savefig(save_path, dpi=100)
        print(f"Plot saved to {save_path}")

    return ax, origin_x, origin_y, legend_elements

def plot_single_vessel_track(ax, track, origin_x, origin_y, legend_elements, track_id, save_plot=False):
    x = track[:, 0] + origin_x
    y = track[:, 1] + origin_y
    # Plot the track
    c = '#2ca02c'  # Green
    ax.plot(x,y, color=c, linewidth=2)
    ax.plot(x[0],y[0], marker='o', color=c, markersize=10)  # Start point
    legend_elements.append(Line2D([0], [0], color=c, label='Actual track', linewidth=2))
    ax.legend(handles=legend_elements, fontsize=12, loc='upper left')
    # Save plot to file
    if save_plot:
        save_path = make_new_directory()
        if track_id is not None:
            save_name = f"Track_{track_id}"
            save_path = os.path.join(save_path, save_name)
        else:
            now_time = datetime.datetime.now().strftime("%H,%M,%S")
            save_path = os.path.join(save_path, f'Prediction_compared_to_track({now_time}).png')
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

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

def plot_predicted_path(ax, point_list, initial_point, r_c, interations, origin_x, origin_y, legend_elements, save_plot=False):
    plot_bouns = False
    if plot_bouns:
        vertecis = [[-110, -110], [-80, -130], [-30, -115], [0, -120], [0, -90], [40, -60], [60, -50], [90, -32], [80, -20], [70, -10], [40, -8], [-20, -6], [-40, -25], [-52, -58], [-60, -68], [-110, -110]]
        for vertex in vertecis:
            vertex[0] += origin_x
            vertex[1] += origin_y
        for i in range(len(vertecis)-1):
            ax.plot([vertecis[i][0], vertecis[i+1][0]], [vertecis[i][1], vertecis[i+1][1]], color='black')
    inital_angle = np.arctan2((initial_point[2]-initial_point[0]),(initial_point[3]-initial_point[1]))*180/np.pi

    c = '#ff7f0e'  # Orange
    point_array = np.array(point_list)
    xs = point_array[:, 0] + origin_x
    ys = point_array[:, 1] + origin_y
    ax.plot(xs, ys, color=c, linewidth=2)
    ax.plot(initial_point[0] + origin_x, initial_point[1] + origin_y, marker='o', color=c, markersize=10)
    quiver = ax.quiver(initial_point[0] + origin_x, initial_point[1] + origin_y, 3*np.sin(np.radians(inital_angle)), 3*np.cos(np.radians(inital_angle)), color='black', scale=5, scale_units='inches', width=0.005)
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label="Initial point" ,markerfacecolor=c, markersize=10))
    legend_elements.append(Line2D([0], [0], color=c, label='Predicted path', linewidth=2))
     # Create a custom legend entry for the quiver
    quiver_key = LineCollection([[(0, 0)]], colors=['black'], label='Initial direction', linewidths=[2])
    legend_elements.append(quiver_key)

    ax.legend(handles=legend_elements, fontsize=12, loc='upper left')

    # Adding useful information to the plot
    text = f"Initial point: [{initial_point[0]:.1f},{initial_point[1]:.1f}], Initial angle: {inital_angle:.1f}, "
    text += f"Search radius: {r_c}, Iterations: {interations}"
    ax.text(0.0, -0.13, text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='grey', alpha=0.15))
    plt.tight_layout() 
    
    # Save plot to file
    if save_plot:
        save_path = make_new_directory()
        now_time = datetime.datetime.now().strftime("%H,%M,%S")
        save_path = os.path.join(save_path, f'plot_predicted_path({now_time}).png')
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    return ax, origin_x, origin_y, legend_elements

def occupancy_grid_to_map(ax):
    # Plotting the occupancy grid'
    data = np.load(f"npy_files/occupancy_grid_without_dilating.npy",allow_pickle='TRUE').item()
    occupancy_grid = data["occupancy_grid"]
    origin_x = data["origin_x"]
    origin_y = data["origin_y"]

    colors = [(1, 1, 1), (0.8, 0.8, 0.8)]  # Black to light gray
    cm = LinearSegmentedColormap.from_list('custom_gray', colors, N=256)
    ax.imshow(occupancy_grid, cmap=cm, interpolation='none', origin='upper', extent=[0, occupancy_grid.shape[1], 0, occupancy_grid.shape[0]])
    
    ax.set_xlim(origin_x-120,origin_x + 120)
    ax.set_ylim(origin_y-140, origin_y + 20)
    ax.set_aspect('equal')
    ax.set_xlabel('East [m]',fontsize=15)
    ax.set_ylabel('North [m]',fontsize=15)
    ax.grid(True)
    return ax, origin_x, origin_y

def plot_histogram(data, gmm, pred_path,track_id, save_plot=False):
    fig2, (ax1,ax2) = plt.subplots(1,2,figsize=(11, 7.166666))

    # plot the predicted path
    ax1, origin_x, origin_y = occupancy_grid_to_map(ax1)
    point_array = np.array(pred_path)
    xs = point_array[:, 0] + origin_x
    ys = point_array[:, 1] + origin_y
    ax1.plot(xs, ys, color='black', linewidth=2)
    ax1.plot(point_array[0, 0] + origin_x, point_array[0, 1] + origin_y, marker='o', color='black', markersize=10)


    # plot the histogram
    ax2.hist(data, bins=100, density=True, alpha=0.6, color='g')
    x = np.linspace(-180, 180, 1000)
    for i in range(gmm.n_components):
        ax2.plot(x, norm.pdf(x, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0])),
                label=f'Component {i+1}')


    ax2.set_xlabel('Course [degrees]', fontsize=15)
    ax2.set_ylabel('Density', fontsize=15)
    ax2.legend(['Data', 'Predicted'], fontsize=12)

    means = gmm.means_
    probs = gmm.weights_
    means = np.array(means).reshape(-1)
    txt_means = ""
    txt_prob = ""
    for (mean,prob) in zip(means,probs):
        if prob < 0.1:
            continue
        txt_means += f"{mean:.1f}, "
        txt_prob += f"{prob:.3f}, "


    # Adding useful information to the plot
    text = f"{gmm.n_components} components, {len(data)} samples\n"
    text += f"Means: {txt_means}\n"
    text += f"Probs: {txt_prob}"
    ax1.text(0.0, -0.3, text, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='grey', alpha=0.15))
    plt.tight_layout() 

    plt.tight_layout()
    # Save plot to file
    if save_plot:
        if track_id is not None:
            dir_name = f"Histograms/Track_{track_id}"
        else:
            dir_name = f"Histograms/no_track_id"
        save_path = make_new_directory(dir_name=dir_name,include_date=False)
        now_time = datetime.datetime.now().strftime("%H,%M,%S")
        save_path = os.path.join(save_path, f'Histogram_({now_time}).png')
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    # plt.show()
    plt.close(fig2)
    # temp_in = input("Press enter to continue")

