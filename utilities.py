import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import os
import datetime
from sklearn.mixture import GaussianMixture
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import re
from plotting import plot_RMSE, plot_random_start_areas

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
        plt.savefig(save_path, dpi=400)
        print(f"Plot saved to {save_path}")

    return ax, origin_x, origin_y, legend_elements

def generate_random_point_and_angle_in_polygon(area, X_B=None, plot=False):
    polygonA = np.array([[60, -50], [90, -30], [45, -8], [30, -16]])
    polygonB = np.array([[-18,-38], [0, -20], [-20, -12], [-30, -25]])
    polygonC = np.array([[-103, -107], [-80, -95], [-60, -110], [-80, -130]])
    polygonD = np.array([[-28,-116], [-5, -116], [-5, -90], [-28, -90]])
    polygonE = np.array([[-50, -58], [-32, -58], [-32, -35], [-50, -35]])
    polygonF = np.array([[4, -25], [28, -25], [28, -8], [4, -8]])
    polygons = [polygonA, polygonB, polygonC, polygonD, polygonE, polygonF]

    polygons_dict = {
        "C": (polygonC, 20, 110),
        "A": (polygonA, 190, 280),
        "D": (polygonD, -40, 40),
        "B": (polygonB, 100, 210),
        "E": (polygonE, 60, 180),
        "F": (polygonF, 110, 270)
    }
    poly = polygons_dict[area]
    polygon = poly[0]
    angle_min = poly[1]
    angle_max = poly[2]

    path = Path(polygon)
    min_x, min_y = np.min(polygon, axis=0)
    max_x, max_y = np.max(polygon, axis=0)

    run_random_point = True
    while run_random_point:
        random_point = np.random.uniform([min_x, min_y], [max_x, max_y])
        if path.contains_point(random_point):
            run_random_point = False
            # return random_point, polygon

    random_angle = np.random.uniform(angle_min, angle_max)
    
    if plot and X_B is not None:
        plot_random_start_areas(X_B, polygons, polygons_dict, random_point, random_angle, area, example_plot=True)
        

    return random_point, random_angle

def make_new_directory(dir_name="Results", include_date=True):
    wokring_directory = os.getcwd()
    root = os.path.join(wokring_directory, dir_name)
    # print(f"Root directory: {root}")

    if not os.path.exists(root):
        os.mkdir(root)
        # print(f"Directory {root} created")
    # else:
    #     print(f"Directory {root} already exists")

    if include_date:
        todays_date = datetime.datetime.now().strftime("%d-%b")
        path = os.path.join(root,todays_date)
        if not os.path.exists(path):
            os.mkdir(path)
    else:
        path = root
    return path

def check_point_within_bounds(point, plot=False):
    """Check if the point is within the bounds of the map."""
    x, y = point[0], point[1]
    vertecis = [[-110, -110], [-80, -130], [-30, -115], [0, -120], [0, -90], [40, -60], [60, -50], [90, -32], [80, -20], [70, -10], [40, -8], [-20, -6], [-40, -25], [-52, -58], [-60, -68], [-110, -110]]
    
    if plot:
        plt.figure()
        plt.plot([vertex[0] for vertex in vertecis], [vertex[1] for vertex in vertecis], 'ro-')
        plt.plot(x, y, 'bo')
        plt.show()

    # check if inside the polygon
    n = len(vertecis)
    inside = False
    p1x, p1y = vertecis[0]
    for i in range(1, n+1):
        p2x, p2y = vertecis[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    # Return True if point is inside the polygon
    # Return False if point is outside the polygon
    return inside

def calculate_course(p1, p2):
    """Calculate the course from point p1 to p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dx, dy))

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def eucldian_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    # Check dimensionality of the points
    # both must have len(4) 
    if len(p1) != 4:
        raise ValueError("Predicted point does not have len of 4.")
    if len(p2) != 4:
        raise ValueError("Point from sub-trajectory does not have len of 4.")
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2 + (p1[3] - p2[3])**2)

def find_initial_points(point,angle):
    """Find the initial points based on the given point and angle."""
    x1 = point[0]
    y1 = point[1]
    x2 = x1 + 4*np.sin(np.radians(angle))
    y2 = y1 + 4*np.cos(np.radians(angle))
    return [x1,y1,x2,y2]

def predict_next_point(average_course, average_distance, current_point):
    """Predict the next point based on average course and distance."""
    x2, y2 = current_point[2], current_point[3]
    x3 = x2 + average_distance * np.sin(np.radians(average_course))
    y3 = y2 + average_distance * np.cos(np.radians(average_course))
    return [x2, y2, x3, y3]

def filter_by_course(neighbours, current_course, delta_course):
    filtered_neighbours = {}
    for track_id, tracks in neighbours.items():
        for track in tracks:
            neighbour_course = calculate_course(track[2:4], track[4:6])
            if abs(neighbour_course - current_course) < delta_course:
                filtered_neighbours.setdefault(track_id, []).append(track)
    return filtered_neighbours

def choice_of_number_of_components(data):
    # Compute BIC to determine the best number of components
    bics = []
    n_components_range = range(1, 9)  # Assuming up to 8 components
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components).fit(data)
        bics.append(gmm.bic(data))
    best_n = n_components_range[np.argmin(bics)]
    return best_n

def constant_velocity_model(point):
    T = 1
    sigma_a = 0.1
    Q = sigma_a ** 2 * np.array(
        [[(T ** 4) / 3, (T ** 3) / 2, 0, 0], [(T ** 3) / 2, T ** 2, 0, 0], [0, 0, (T ** 4) / 3, (T ** 3) / 2],
            [0, 0, (T ** 3) / 2, T ** 2]]
    )
    F = np.array([[1, T, 0, 0], [0, 1, 0, 0], [0, 0, 1, T], [0, 0, 0, 1]])
    t = 1.0
    current_state = np.array([point[2], (1 / t) * (point[2] - point[0]), point[3], (1 / t) * (point[3] - point[1])])
    predicted_state = np.dot(F, current_state) + np.random.multivariate_normal([0, 0, 0, 0], Q)
    sub_trajectory = [point[0], point[1], point[2], point[3], predicted_state[0], predicted_state[2]]
    return sub_trajectory

def add_CVM(closest_neighbours, point):
    alpha = 100 
    M = len(closest_neighbours)
    if M == 0:
        W = 1
    else:
        W = round(alpha / M)
    for i in range(W):
        closest_neighbours.setdefault(-1, []).append(constant_velocity_model(point))
    return closest_neighbours

def check_similarites_between_tracks(path1,path2):
    path1 = np.array(path1)
    path2 = np.array(path2[1])

    # distance_count = 0
    # for (point1,point2) in zip(path1,path2):
    #     distance = np.linalg.norm(point1[2:] - point2[2:])
    #     distance_count += distance
    # # print(f"Total distance between the paths: {distance_count}")
    # if distance_count < 20:
    #     return True
    # else:
    #     return False

    new_point = path1[-1][2:]
    # Check endpoint
    for point in path2:
        distance = np.linalg.norm(new_point - point[2:])
        if distance < 1:
            return True
    return False
    # if distance < 5:
    #     return True
    # else:
    #     return False

def check_similarites_between_tracks_2(path1,path2):

    path1 = np.array(path1)
    path2 = np.array(path2)

    if len(path1) != len(path2):
        return False
    
    for (point1,point2) in zip(path1,path2):
        distance = np.linalg.norm(point1[2:] - point2[2:])
        if distance > 5:
            return False
    return True

    # print("Length of path1 and path2")
    # print(len(path1),len(path2))
    # return True

def RMSE(original_track, predicted_track, plot_statement=False, save_plot=False):

    # Calculate the distance between the last point of the original track and each point in the predicted track
    distances = np.linalg.norm(original_track[-1] - predicted_track, axis=1)

    # Find the index of the point in the predicted track that is closest to the end point of the original track
    index_to_cut = np.argmin(distances)

    if index_to_cut < 5:
        # If the distance exceeds the threshold, do not cut the predicted track
        index_to_cut = len(predicted_track) - 1
    
    # Cut the predicted track short
    predicted_track_cut = predicted_track[:index_to_cut+1]

    # Interpolate predicted track to fit original track
    interp_func = interp1d(np.linspace(0, 1, len(predicted_track_cut)), predicted_track_cut, axis=0)
    interpolated_predicted_track = interp_func(np.linspace(0, 1, len(original_track)))

    if plot_statement:
        original_track = np.array(original_track)
        predicted_track = np.array(predicted_track)
        interpolated_predicted_track = np.array(interpolated_predicted_track)
        plot_RMSE(original_track, predicted_track, interpolated_predicted_track,save_plot=save_plot)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(original_track, interpolated_predicted_track))
    return rmse

def read_results():
    # Define variables to store the sum of RMSE values for r_c=3 and r_c=10
    total_rmse_rc3 = 0
    total_rmse_rc10 = 0
    count = 0
    # Open the text file
    with open("results.txt", "r") as file:
        # Read each line
        for line in file:
            line = line.split(":")
            value1 = line[2].split(",")[0]
            value2 = line[3].split("\n")[0]

            # Extract RMSE values for r_c=3 and r_c=10
            rmse_rc3 = float(value1)
            rmse_rc10 = float(value2)
            # print(f"RMSE for r_c=3: {rmse_rc3}, RMSE for r_c=10: {rmse_rc10}")

            # Add RMSE values to the total
            total_rmse_rc3 += rmse_rc3
            total_rmse_rc10 += rmse_rc10
            count += 1


    # Print the total RMSE for r_c=3 and r_c=10
    print(f"Total RMSE for r_c=3: {total_rmse_rc3/count:.2f}")
    print(f"Total RMSE for r_c=10: {total_rmse_rc10/count:.2f}")

def compare_different_rc(track_id, rmse, rc, number_of_close_neighbor_list ,reset=False):
    if reset:
        with open("comparison_of_rc.txt", "w") as file:
            file.write("")
    
    number_of_close_neighbor_list = np.array(number_of_close_neighbor_list)
    avg_number_of_close_neighbor = np.mean(number_of_close_neighbor_list)

    with open("comparison_of_rc.txt", "r") as file:
        lines = file.readlines()
        if len(lines) > 1:
            last_id = int(lines[-1].split(":")[1].split(",")[0])
        else:
            last_id = track_id

    if last_id != track_id:
        with open("comparison_of_rc.txt", "a") as file:
            file.write("\n")
            file.write(f"Track_id: {track_id}, rc: {rc}, CN: {avg_number_of_close_neighbor:.2f} RMSE: {rmse:.2f}\n")
    else:
        with open("comparison_of_rc.txt", "a") as file:
            file.write(f"Track_id: {track_id}, rc: {rc}, CN: {avg_number_of_close_neighbor:.2f} RMSE: {rmse:.2f}\n")

def find_best_rc():
    # Define a dictionary to store the best rc for each track
    best_rc = {}

    # Open the text file
    with open('comparison_of_rc.txt', 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line into components
            if line == "\n":
                continue
            info = line.split()
            track_id = int(info[1].split(",")[0])
            rc = int(info[3].split(",")[0])
            CN = float(info[5].split(",")[0])
            RMSE = float(info[7].split(",")[0])

            # Check if the track_id already exists in the best_rc dictionary
            if track_id in best_rc:
                # If the current RMSE is lower than the previous best RMSE for the same track,
                # update the best_rc dictionary with the current rc
                if RMSE < best_rc[track_id]['RMSE']:
                    best_rc[track_id] = {'rc': rc, 'RMSE': RMSE}
            else:
                # If the track_id is not in the dictionary, add it with the current rc and RMSE
                best_rc[track_id] = {'rc': rc, 'RMSE': RMSE}

    # Print the best rc for each track
    for track_id, data in best_rc.items():
        print(f"Track_id: {track_id}, Best rc: {data['rc']}, RMSE: {data['RMSE']}")


