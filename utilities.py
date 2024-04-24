import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os
import datetime

def generate_random_point_and_angle_in_polygon(area, plot=False):
    if area == "west":
        polygon = np.array([[-100, -120], [-100, -100], [-80, -100], [-80, -120]])
        angle_min = 30
        angle_max = 110
    elif area == "east":
        polygon = np.array([[60, -50], [90, -30], [42, -15], [35, -18]])
        angle_min = 190
        angle_max = 260
    elif area == "south":
        polygon = np.array([[-28,-120], [-10, -120], [-10, -90], [-28, -90]])
        angle_min = -40
        angle_max = 40
    elif area == "north":
        polygon = np.array([[-18,-38], [0, -20], [-20, -12], [-30, -25]])
        angle_min = 100
        angle_max = 210
    else:
        raise ValueError('Invalid area. Choose either "west", "east", "south" or "north"')
    
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
        plt.figure()
        plt.fill(polygon[:, 0], polygon[:, 1], 'b', alpha=0.3)  # Polygon filled with blue color
        plt.plot(*random_point, 'ro')  # Random point marked in red
        vector_length = 10
        plt.quiver(random_point[0], random_point[1], vector_length*np.sin(np.radians(random_angle)), vector_length*np.cos(np.radians(random_angle)), color='g')
        # plt.plot(random_point[0] + vector_length*np.cos(np.radians(random_angle)), random_point[1] + vector_length*np.sin(np.radians(random_angle)), 'go')  # Random angle marked in green
        plt.grid(True)
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
    vertecis = [[-110, -110], [-80, -130], [-30, -115], [0, -120], [0, -90], [40, -60], [60, -50], [95, -20], [80, -10], [40, -8], [-20, -6], [-40, -25], [-52, -58], [-60, -68], [-110, -110]]
    
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







