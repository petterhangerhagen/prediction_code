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
from sklearn.mixture import GaussianMixture
from GMM_components import get_GMM_modified

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

def plot_all_vessel_tracks(ax, X_B, origin_x, origin_y, save_plot=False, marker_size=0.01):

    # Extract points for plotting
    points = [X_B[key][:, 0:2] for key in X_B]
    points = np.concatenate(points, axis=0)  # combine all track points into a single array
    xs = np.array(points[:, 0]) + origin_x  # shift x-coordinates by origin_x
    ys = np.array(points[:, 1]) + origin_y  # shift y-coordinates by origin_y
    scatter = ax.scatter(xs, ys, s=marker_size, marker=".", color='#1f77b4')

    # Create a legend with a larger marker
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='All tracks',
                            markerfacecolor=scatter.get_facecolor()[0], markersize=10)]
    # ax.legend(handles=legend_elements, fontsize=12, loc='upper left')

    # Save plot to file
    if save_plot:
        save_path = 'Images/plot_all_vessel_tracks.png'
        plt.savefig(save_path, dpi=100)
        print(f"Plot saved to {save_path}")

    return ax, origin_x, origin_y, legend_elements

def plot_single_vessel_track(ax, track, origin_x, origin_y, legend_elements, track_id, save_plot=False):
    track = np.array(track)
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
            now_time = datetime.datetime.now().strftime("%H,%M,%S")
            save_name = f"Track_{track_id}_plot({now_time}).png"
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

def plot_predicted_path(ax, point_list, initial_point, r_c, interations,rmse, origin_x, origin_y, legend_elements, save_plot=False):
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
    text += f"Search radius: {r_c}, Iterations: {interations}, RMSE: {rmse:.3f}\n"
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

    return ax, origin_x, origin_y

def plot_histogram(data, gmm, pred_path,track_id, sim=None, weight=None, save_plot=False):
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
    x = np.linspace(-360, 360, 1000)
    for i in range(gmm.n_components):
        ax2.plot(x, norm.pdf(x, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0])),
                label=f'Component {i+1}')


    ax2.set_xlabel('Course [degrees]', fontsize=15)
    ax2.set_ylabel('Density', fontsize=15)
    ax2.legend(fontsize=12)

    means = gmm.means_
    probs = gmm.weights_
    means = np.array(means).reshape(-1)
    txt_means = ""
    txt_prob = ""
    for (mean,prob) in zip(means,probs):
        # if prob < 0.1:
        #     continue
        txt_means += f"{mean:.1f}, "
        txt_prob += f"{prob:.3f}, "

    if sim is not None:
        text_sim = "Sim: "
        for (s,prob) in zip(sim,probs):
            # if prob < 0.1:
            #     continue
            text_sim += f"{s:.3f}, "
    else:
        text_sim = ""

    if weight is not None:
        text_weight = "W-score: "
        for (w,prob) in zip(weight,probs):
            # if prob < 0.1:
            #     continue
            text_weight += f"{w:.3f}, "
    else:
        text_weight = ""


    # Adding useful information to the plot
    text = f"{gmm.n_components} components, {len(data)} samples\n"
    text += f"Means: {txt_means}\n"
    text += f"Probs: {txt_prob}\n"
    text += f"{text_sim}\n"
    text += f"{text_weight}"
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

def plot_close_neigbors(neighbors):
    fig, ax = plt.subplots(figsize=(11, 7.166666))
    count = 0
    print(f"len(neighbours): {len(neighbors)}")
    for key,value in neighbors.items():
        for sub_track in value:
            # print(sub_track)
            # print([sub_track[0],sub_track[2],sub_track[4]])
            course = calculate_course(sub_track[2:4],sub_track[4:6])
            
            if -10 < course < 30:
                x = [sub_track[0],sub_track[2],sub_track[4]]
                y = [sub_track[1],sub_track[3],sub_track[5]]
                start = [sub_track[0],sub_track[1]]
                ax.plot(x,y)
                ax.scatter(x,y)
                ax.plot(start[0],start[1], marker='o', color='black', markersize=10)

                count += 1
    print(f"Number of close neighbors: {len(neighbors)}")
    print(f"Count: {count}")
    plt.show()
    plt.close(fig)

def calculate_course(p1, p2):
    """Calculate the course from point p1 to p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dx, dy))

def plot_recursive_paths(npy_file):

    data = np.load(npy_file,allow_pickle='TRUE').item()
    
    for key, value in data.items():
        fig, ax = plt.subplots(figsize=(11, 7.166666))
        ax, origin_x, origin_y = occupancy_grid_to_map(ax)
        point_list = value
        point_array = np.array(point_list)
        xs = point_array[:, 0] + origin_x
        ys = point_array[:, 1] + origin_y
        ax.plot(xs, ys, linewidth=2)
        # ax.plot(point_array[0, 0], point_array[0, 1], marker='o', markersize=10)
    plt.show()
    plt.close(fig)

def plot_histogram_distances(distances, pred_path, X_B, track_id=None, iteration_num=0, save_plot=False):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20, 7.166666))
    font_size = 17
    # Main plot
     # ----------------------------------#
    distances = np.array(distances)
    ax1.hist(distances, density=True, bins=100, color='#1f77b4')
    ax1.set_xlabel('Distances of close neighbors [m]', fontsize=font_size*1.33)
    ax1.set_ylabel('Density', fontsize=font_size*1.33)
    ax1.tick_params(axis='both', which='major', labelsize=font_size*1.33)
    plt.tight_layout()

    GMM = GaussianMixture(n_components=1,random_state=0).fit(distances.reshape(-1, 1))
    x = np.linspace(0, np.max(distances), 1000)
    GMM_plot = ax1.plot(x, norm.pdf(x, GMM.means_[0, 0], np.sqrt(GMM.covariances_[0, 0, 0])), lw="3", label='GMM', color='#c73838')
    
    # Find the maximum point of the PDF
    max_pdf_point = GMM.means_[0, 0]
    max_pdf_value = norm.pdf(max_pdf_point, GMM.means_[0, 0], np.sqrt(GMM.covariances_[0, 0, 0]))
    prob = GMM.weights_[0]
    x_pos = max_pdf_point
    y_pos = max_pdf_value
    ax1.text(x_pos*1.6, y_pos*1.03, f"\u03BC = {max_pdf_point:.2f}\np(x)={prob:.4f}", fontsize=font_size*1.1, verticalalignment='bottom', horizontalalignment='center')
    
    # Plot a dot at the top of the distribution
    mean_plot = ax1.plot(max_pdf_point, max_pdf_value, color='#c73838', marker='o', markersize=10, label='GMM mean')
 
    legend_elements_1 = [Line2D([0], [0], color='#c73838', label='GMM', linewidth=2),
                        Line2D([0], [0], marker='o', color='w', label='GMM means', markerfacecolor='#c73838', markersize=10)]
    ax1.legend(handles=legend_elements_1,loc="upper left",fontsize=font_size)
    # ----------------------------------#

    # plot the predicted path
    ax2, origin_x, origin_y = occupancy_grid_to_map(ax2)
    ax2, origin_x, origin_y, legend_elements = plot_all_vessel_tracks(ax2, X_B, origin_x, origin_y, marker_size=0.01)
    color = '#ff7f0e'  # Orange
    point_array = np.array(pred_path)
    xs = point_array[:, 0] + origin_x
    ys = point_array[:, 1] + origin_y
    ax2.plot(xs, ys, color=color, linewidth=2)
    ax2.plot(point_array[0, 0] + origin_x, point_array[0, 1] + origin_y, marker='o', color=color, markersize=15)
    ax2.plot(point_array[-1, 0] + origin_x, point_array[-1, 1] + origin_y, marker='o', color='black', markersize=10)
    # Remove x_label and y_label
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    # ax2.tick_params(axis='both', which='major', labelsize=12)
    legend_elements_2 = [Line2D([0], [0], color=color, label='Predicted path', linewidth=2),
                        Line2D([0], [0], marker='o', color="w", label='Predicted path start', markerfacecolor=color, markersize=15),
                        Line2D([0], [0], marker='o', color='w', label='Current prediction point', markerfacecolor='black', markersize=10)]
    ax2.legend(handles=legend_elements_2, fontsize=font_size, loc='lower right')

    ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    # Remove x_label and y_label
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    # Remove numbers on the axis
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    if save_plot:
        save_path = make_new_directory(dir_name=f'Distance_histogram/Track_{track_id}', include_date=False)
        if track_id is not None:
            # now_time = datetime.datetime.now().strftime("%H,%M,%S")
            save_name = f"Track_{track_id}_plot_{iteration_num}.png"
            save_path = os.path.join(save_path, save_name)
        else:
            now_time = datetime.datetime.now().strftime("%H,%M,%S")
            save_path = os.path.join(save_path, f'Distance_histogram({now_time}).png')
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_histogram_courses(courses, num_comp, pred_path, X_B, track_id=None, iteration_num=0, save_plot=False):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20, 7.166666))
    font_size = 17
    # Main plot
     # ----------------------------------#
    courses = np.array(courses)
    ax1.hist(courses, density=True, bins=100, color='#1f77b4')
    ax1.set_xlabel('Courses of close neighbors [\u00B0]', fontsize=font_size*1.33)
    ax1.set_ylabel('Density', fontsize=font_size*1.33)
    ax1.tick_params(axis='both', which='major', labelsize=font_size*1.33)
    plt.tight_layout()



    GMM = get_GMM_modified(courses, 8,60)
    # GMM = GaussianMixture(n_components=num_comp,random_state=0).fit(courses.reshape(-1, 1))
    GMM.fit(courses.reshape(-1, 1))
    x = np.linspace(np.min(courses), np.max(courses), 1000)
    y_poses = []

    for num in range(len(GMM.means_)):
        GMM_plot = ax1.plot(x, norm.pdf(x, GMM.means_[num, 0], np.sqrt(GMM.covariances_[num, 0, 0])), lw="3", color='#c73838')
        max_pdf_point = GMM.means_[num, 0]
        max_pdf_value = norm.pdf(max_pdf_point, GMM.means_[num, 0], np.sqrt(GMM.covariances_[num, 0, 0]))
        mean_plot = ax1.plot(max_pdf_point, max_pdf_value, color='#c73838', marker='o', markersize=10)
        prob = GMM.weights_[num]
        x_pos = max_pdf_point - 10
        y_pos = max_pdf_value #+ 0.0005
        y_poses.append(y_pos)
        diff = 0
        if x_pos < 0:
            diff = abs(x_pos*0.2)
        elif x_pos > 0:
            diff = -abs(x_pos*0.15)
        ax1.text(x_pos + diff , y_pos*1.03, f"\u03BC = {max_pdf_point:.2f}\np(x)={prob:.4f}", fontsize=font_size*1.1, verticalalignment='bottom', horizontalalignment='center')
    # ax1.set_xlim(-240, 240)
    ax1.set_ylim(0,max(y_poses)*1.5)

    add_mean = False
    if add_mean:
        mean_courses = np.mean(courses)
        ax1.plot([mean_courses, mean_courses], [0, 0.005], color='#2ca02c', lw="8")
        ax1.text(mean_courses, 0.006, f"{mean_courses:.2f}", fontsize=font_size*1.1, verticalalignment='bottom', horizontalalignment='center')

        legend_elements_1 = [Line2D([0], [0], color='#c73838', label='GMM', linewidth=2),
                            Line2D([0], [0], marker='o', color='w', label='GMM means', markerfacecolor='#c73838', markersize=10),
                            Line2D([0], [0], color='#2ca02c', label='Average course', linewidth=4)]
    else:
        legend_elements_1 = [Line2D([0], [0], color='#c73838', label='GMM', linewidth=2),
                            Line2D([0], [0], marker='o', color='w', label='GMM means', markerfacecolor='#c73838', markersize=10)]
    ax1.legend(handles=legend_elements_1,loc="upper left",fontsize=font_size)
    # # ----------------------------------#


    # A smaller plot within the main plot
    # left, bottom, width, height = 0.73, 0.71, 0.25, 0.25
    # ax2 = fig.add_axes([left, bottom, width, height])

    # plot the predicted path
    ax2, origin_x, origin_y = occupancy_grid_to_map(ax2)
    ax2, origin_x, origin_y, legend_elements = plot_all_vessel_tracks(ax2, X_B, origin_x, origin_y, marker_size=0.01)
    color = '#ff7f0e'  # Orange
    point_array = np.array(pred_path)
    xs = point_array[:, 0] + origin_x
    ys = point_array[:, 1] + origin_y
    ax2.plot(xs, ys, color=color, linewidth=2)
    ax2.plot(point_array[0, 0] + origin_x, point_array[0, 1] + origin_y, marker='o', color=color, markersize=15)
    ax2.plot(point_array[-1, 0] + origin_x, point_array[-1, 1] + origin_y, marker='o', color='black', markersize=10)
    # Remove x_label and y_label
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    # ax2.tick_params(axis='both', which='major', labelsize=12)
    legend_elements_2 = [Line2D([0], [0], color=color, label='Predicted path', linewidth=2),
                        Line2D([0], [0], marker='o', color="w", label='Predicted path start', markerfacecolor=color, markersize=15),
                        Line2D([0], [0], marker='o', color='w', label='Current prediction point', markerfacecolor='black', markersize=10)]
    ax2.legend(handles=legend_elements_2, fontsize=font_size, loc='lower right')

    ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    # Remove x_label and y_label
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    # Remove numbers on the axis
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    if save_plot:
        save_path = make_new_directory(dir_name=f'Courses_histogram/Track_{track_id}', include_date=False)
        if track_id is not None:
            # now_time = datetime.datetime.now().strftime("%H,%M,%S")
            if np.max(courses) > 190:
                save_name = f"Track_{track_id}_plot_{iteration_num}_1.png"
            else:
                save_name = f"Track_{track_id}_plot_{iteration_num}_2.png"
            save_path = os.path.join(save_path, save_name)
        else:
            now_time = datetime.datetime.now().strftime("%H,%M,%S")
            save_path = os.path.join(save_path, f'course_histogram({now_time}).png')
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    if not save_plot:
        plt.show()
    plt.close(fig)


def gmm_plot(data, gmm, prev_gmm):
    fig, (ax,ax2) = plt.subplots(1,2,figsize=(17, 7.166666))
    ax.hist(data, bins=100, density=True, alpha=0.6, color='g')
    ax2.hist(data, bins=100, density=True, alpha=0.6, color='g')
    x = np.linspace(-360, 360, 1000)
    for i in range(gmm.n_components):
        ax.plot(x, norm.pdf(x, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0])),"blue")

        max_pdf_point = gmm.means_[i, 0]
        max_pdf_value = norm.pdf(max_pdf_point, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0]))
        mean_plot = ax.plot(max_pdf_point, max_pdf_value, color='blue', marker='o', markersize=10)
        prob = gmm.weights_[i]
        x_pos = max_pdf_point - 10
        y_pos = max_pdf_value #+ 0.0005
        # y_poses.append(y_pos)
        ax.text(x_pos , y_pos, f"\u03BC = {max_pdf_point:.2f}\np(x)={prob:.4f}", fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    
    for i in range(prev_gmm.n_components):
        ax2.plot(x, norm.pdf(x, prev_gmm.means_[i, 0], np.sqrt(prev_gmm.covariances_[i, 0, 0])),"red")
        max_pdf_point = prev_gmm.means_[i, 0]
        max_pdf_value = norm.pdf(max_pdf_point, prev_gmm.means_[i, 0], np.sqrt(prev_gmm.covariances_[i, 0, 0]))
        ax2.plot(max_pdf_point, max_pdf_value, color='red', marker='o', markersize=10)
        prob = prev_gmm.weights_[i]
        x_pos = max_pdf_point - 10
        y_pos = max_pdf_value #+ 0.0005
        # y_poses.append(y_pos)
        ax2.text(x_pos , y_pos, f"\u03BC = {max_pdf_point:.2f}\np(x)={prob:.4f}", fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    
       
    ax.set_xlabel('Course [degrees]')
    ax.set_ylabel('Density')
    ax2.set_xlabel('Course [degrees]')
    ax2.set_ylabel('Density')
    plt.show()