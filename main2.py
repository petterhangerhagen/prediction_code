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
    plt.savefig(save_path, dpi=400)
    print(f"Plot saved to {save_path}")

    return ax, origin_x, origin_y, legend_elements

def plot_single_vessel_track(ax, track_id, X_B, origin_x, origin_y):
    """Plot a single vessel track identified by track_id.
    
    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        track_id (int): The ID of the track to plot.
        X_B (dict): Dictionary of tracks.
    """
    track = X_B[track_id]
    points = track[:, 0:2]

    # Plot the track
    ax.plot(points[:, 0], points[:, 1], color='red')
    ax.plot(points[0, 0], points[0, 1], marker='o', color='green')  # Start point
    ax.set_xlim(-120, 120)
    ax.set_ylim(-140, 20)
    
    # Save plot to file
    save_path = 'Images/plot_single_vessel_track.png'
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

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

def calculate_course(p1, p2):
    """Calculate the course from point p1 to p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def find_closest_neighbours(X_B, point, radius):
    """Find all neighbours within the 'radius' from 'point'.
    
    Args:
        X_B (dict): Dictionary of all tracks.
        point (list): The reference point [x1, y1, x2, y2].
        radius (float): The radius within which to search for neighbours.
    """
    target_point = point[2:4]
    closest_neighbours = {}
    for track_id, track_points in X_B.items():
        for sub_track in track_points:
            if all(calculate_distance(target_point, sub_track[i:i+2]) <= radius for i in range(0, len(sub_track), 2)):
                closest_neighbours.setdefault(track_id, []).append(sub_track)
    return closest_neighbours

def filter_by_course(neighbours, point, delta_course):
    """Filter neighbours that are within 'delta_course' of the initial course defined by 'point'.
    
    Args:
        neighbours (dict): Neighbours to filter.
        point (list): Initial point [x1, y1, x2, y2].
        delta_course (float): Course deviation allowed from the initial course.
    """
    initial_course = calculate_course(point[:2], point[2:4])
    filtered_neighbours = {}
    for track_id, tracks in neighbours.items():
        for track in tracks:
            neighbour_course = calculate_course(track[2:4], track[4:6])
            if abs(neighbour_course - initial_course) < delta_course:
                filtered_neighbours.setdefault(track_id, []).append(track)
    return filtered_neighbours

def compute_average_course_and_distance(neighbours):
    """Compute the average course and distance for filtered neighbour tracks."""
    total_course = total_distance = count = 0
    for tracks in neighbours.values():
        for track in tracks:
            course = calculate_course(track[2:4], track[4:6])
            distance = calculate_distance(track[2:4], track[4:6])
            total_course += course
            total_distance += distance
            count += 1
    average_course = total_course / count
    average_distance = total_distance / count
    return average_course, average_distance

def predict_next_point(average_course, average_distance, current_point):
    """Predict the next point based on average course and distance."""
    x2, y2 = current_point[2], current_point[3]
    x3 = x2 + average_distance * np.cos(np.radians(average_course))
    y3 = y2 + average_distance * np.sin(np.radians(average_course))
    return [x2, y2, x3, y3]

def iterative_path_prediction(X_B):
    """Iteratively predict path based on initial point and movement statistics."""
    initial_point = [65.7, -35.3, 61.8, -36.5]  # Example specific coordinates
    r_c = 10
    delta_course = 15
    K = 100  # Total iterations

    point_list = [initial_point[:2]]
    current_point = initial_point

    for k in range(K):
        neighbours = find_closest_neighbours(X_B, current_point, r_c)
        if not neighbours:
            break
        within_course = filter_by_course(neighbours, current_point, delta_course)
        if not within_course:
            break
        avg_course, avg_distance = compute_average_course_and_distance(within_course)
        current_point = predict_next_point(avg_course, avg_distance, current_point)
        point_list.append(current_point[:2])

    point_list.append(current_point[2:4])
    return point_list

if __name__ == '__main__':
    X_B = np.load('npy_files/X_B.npy', allow_pickle=True).item()
    ax, origin_x, origin_y, legend_elements = plot_all_vessel_tracks(X_B)

    # plot_single_vessel_track(ax, list(X_B.keys())[0], X_B)  # example: plot first track

    pred_path = iterative_path_prediction(X_B)
    plot_predicted_path(ax, pred_path, origin_x, origin_y, legend_elements)
