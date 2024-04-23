import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_all_vessel_tracks, plot_single_vessel_track, plot_predicted_path
from sklearn.mixture import GaussianMixture


def calculate_course(p1, p2):
    """Calculate the course from point p1 to p2."""
    # dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dx, dy))

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

def find_initial_points(point,angle):
    """Find the initial points based on the given point and angle."""
    x1 = point[0]
    y1 = point[1]
    x2 = x1 + 1*np.sin(np.radians(angle))
    y2 = y1 + 1*np.cos(np.radians(angle))
    return [x1,y1,x2,y2]


# def compute_average_course_and_distance(neighbours):
#     """Compute the average course and distance for filtered neighbour tracks."""
#     total_course = total_distance = count = 0
#     for tracks in neighbours.values():
#         for track in tracks:
#             course = calculate_course(track[2:4], track[4:6])
#             distance = calculate_distance(track[2:4], track[4:6])
#             total_course += course
#             total_distance += distance
#             count += 1
#     average_course = total_course / count
#     average_distance = total_distance / count
#     return average_course, average_distance

def compute_average_course_and_distance(neighbours, plot_statement = False):
    total_course = total_distance = count = 0
    courses = []  # List to store all course values
    for tracks in neighbours.values():
        for track in tracks:
            course = calculate_course(track[2:4], track[4:6])
            distance = calculate_distance(track[2:4], track[4:6])
            total_course += course
            total_distance += distance
            count += 1
            courses.append(course)  # Add the course to the list
    average_course = total_course / count
    average_distance = total_distance / count

    print(f'Average course: {average_course:.2f}')
    if plot_statement:
        # Plot a histogram of the courses
        fig2, ax2 = plt.subplots(figsize=(11, 7.166666))
        ax2.hist(courses, bins=30,density=False, alpha=0.75, color='b', edgecolor='black')
        ax2.set_xlabel('Course')
        ax2.set_ylabel('Probability')
        ax2.set_title('Probability Distribution of Courses')
        plt.show()

    return average_course, average_distance


def predict_next_point(average_course, average_distance, current_point):
    """Predict the next point based on average course and distance."""
    x2, y2 = current_point[2], current_point[3]
    x3 = x2 + average_distance * np.sin(np.radians(average_course))
    y3 = y2 + average_distance * np.cos(np.radians(average_course))
    return [x2, y2, x3, y3]


def iterative_path_prediction(initial_point, r_c, delta_course, K, X_B):
    """Iteratively predict path based on initial point and movement statistics."""
    

    point_list = [initial_point[:2]]
    current_point = initial_point

    for k in range(K):
        print(f'Iteration {k}')
        neighbours = find_closest_neighbours(X_B, current_point, r_c)
        if not neighbours:
            break
        within_course = filter_by_course(neighbours, current_point, delta_course)
        if not within_course:
            break
        if k > 20:
            compute_average_course_and_distance(within_course, plot_statement=False)
        else:
            avg_course, avg_distance = compute_average_course_and_distance(within_course)
        print(f'Average course: {avg_course:.2f}, Average distance: {avg_distance:.2f}')
        current_point = predict_next_point(avg_course, avg_distance, current_point)
        print(f'Next point: {current_point}')
        point_list.append(current_point[:2])
        print("\n")

    point_list.append(current_point[2:4])
    return point_list


def main():
    # Load the data
    X_B = np.load('npy_files/X_B.npy', allow_pickle=True).item()

    # Define the initial point and angle
    # initial_point = [65.7, -35.3]
    # initial_point = [65.7, -20]

    # initial_point = [-30, -95]
    # angle = -70

    # initial_point = [-18, -15]
    # angle = 180 + 5

    initial_point = [-10, -110]
    angle = -5

    initial_point = find_initial_points(initial_point,angle)

    # Parameters for iterative path prediction
    r_c = 10
    delta_course = 25
    K = 30 # Total iterations

    # Run the iterative path prediction algorithm
    pred_paths = iterative_path_prediction(initial_point, r_c, delta_course, K, X_B)

    # Plot the predicted path
    ax, origin_x, origin_y, legend_elements = plot_all_vessel_tracks(X_B)
    plot_predicted_path(ax, pred_paths, origin_x, origin_y, legend_elements) 
    plt.show() 

if __name__ == '__main__':
    main()
