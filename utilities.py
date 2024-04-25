import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import os
import datetime

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

def generate_random_point_and_angle_in_polygon(area, X_B, plot=False):
    # polygonC = np.array([[-100, -120], [-100, -100], [-80, -100], [-80, -120]])
    polygonA = np.array([[60, -50], [90, -30], [42, -15], [35, -18]])
    polygonB = np.array([[-18,-38], [0, -20], [-20, -12], [-30, -25]])
    polygonC = np.array([[-103, -107], [-80, -95], [-60, -110], [-80, -130]])
    polygonD = np.array([[-28,-116], [-5, -116], [-5, -90], [-28, -90]])
    polygonE = np.array([[-50, -58], [-32, -58], [-32, -35], [-50, -35]])
    polygonF = np.array([[4, -25], [28, -25], [28, -8], [4, -8]])
    polygons = [polygonA, polygonB, polygonC, polygonD, polygonE, polygonF]

    if area == "west" or area == "C":
        polygon = polygonC
        angle_min = 30
        angle_max = 110
    elif area == "east" or area == "A":
        polygon = polygonA
        angle_min = 190
        angle_max = 260
    elif area == "south" or area == "D":
        polygon = polygonD
        angle_min = -40
        angle_max = 40
    elif area == "north" or area == "B":
        polygon = polygonB
        angle_min = 100
        angle_max = 210
    elif area == "E":
        polygon = polygonE
        angle_min = 60
        angle_max = 180
    elif area == "F":
        polygon = polygonF
        angle_min = 110
        angle_max = 270
    else:
        raise ValueError('Invalid area')
    
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
    
    if plot:
        ax1, origin_x, origin_y = start_plot()
        ax1.plot(origin_x, origin_y, 'ro')  # Origin marked in red
        ax1, origin_x, origin_y, legend_elements = plot_all_vessel_tracks(ax1, X_B, origin_x, origin_y,save_plot=False)
        for poly in polygons:
        # Need to shift the polygon to the origin
            temp_poly = np.zeros_like(poly)
            for i in range(len(poly)):
                temp_poly[i][0] = poly[i][0] + origin_x
                temp_poly[i][1] = poly[i][1] + origin_y
            ax1.fill(temp_poly[:, 0], temp_poly[:, 1], 'b', alpha=0.3)  # Polygon filled with blue color
            ax1.plot(temp_poly[:, 0], temp_poly[:, 1], 'b')  # Polygon outline in blue
        random_point_plot = np.zeros_like(random_point)
        random_point_plot[0] = random_point[0] + origin_x
        random_point_plot[1] = random_point[1] + origin_y
        ax1.plot(*random_point_plot, 'ro')  # Random point marked in red
        vector_length = 10
        ax1.quiver(random_point_plot[0], random_point_plot[1], vector_length*np.sin(np.radians(random_angle)), vector_length*np.cos(np.radians(random_angle)), color='g')
        # plt.plot(random_point[0] + vector_length*np.cos(np.radians(random_angle)), random_point[1] + vector_length*np.sin(np.radians(random_angle)), 'go')  # Random angle marked in green
        ax1.grid(True)
        plt.show()
        plt.close()

    return random_point, random_angle

def make_new_directory():
    wokring_directory = os.getcwd()
    root = os.path.join(wokring_directory, "Results")
    print(f"Root directory: {root}")

    if not os.path.exists(root):
        os.mkdir(root)
        print(f"Directory {root} created")
    else:
        print(f"Directory {root} already exists")

    todays_date = datetime.datetime.now().strftime("%d-%b")
    path = os.path.join(root,todays_date)
    if not os.path.exists(path):
        os.mkdir(path)
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
    x2 = x1 + 3*np.sin(np.radians(angle))
    y2 = y1 + 3*np.cos(np.radians(angle))
    return [x1,y1,x2,y2]

def predict_next_point(average_course, average_distance, current_point):
    """Predict the next point based on average course and distance."""
    x2, y2 = current_point[2], current_point[3]
    x3 = x2 + average_distance * np.sin(np.radians(average_course))
    y3 = y2 + average_distance * np.cos(np.radians(average_course))
    return [x2, y2, x3, y3]







