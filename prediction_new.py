import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_all_vessel_tracks, plot_single_vessel_track, plot_predicted_path, plot_predicted_path_new
from sklearn.mixture import GaussianMixture


def calculate_course(p1, p2):
    """Calculate the course from point p1 to p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
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

def compute_probabilistic_course(neighbours):
    courses = []  # List to store all course values
    total_distance = count =  0
    for tracks in neighbours.values():
        for track in tracks:
            course = calculate_course(track[2:4], track[4:6])
            courses.append(course)  # Add the course to the list

            # Calculate the total distance
            total_distance += calculate_distance(track[2:4], track[4:6])
            count += 1
    
    average_distance = total_distance / count
            
    # Reshape courses for GMM
    courses = np.array(courses).reshape(-1, 1)
    print(f"Number of samples: {courses.shape[0]}")

    probabilities_list = []

    if courses.shape[0] < 60:
        predicted_course = [np.mean(courses)]
        probabilities_list.append(1.0)
    else:
        # List to store the BIC values for different numbers of components
        bics = []

        # Range of possible numbers of components
        n_components_range = range(1, 10)

        for n_components in n_components_range:
            # Fit a GMM with n_components
            gmm = GaussianMixture(n_components=n_components).fit(courses)

            # Append the BIC for this model to the list
            bics.append(gmm.bic(courses))

        # Find the number of components that gives the lowest BIC
        best_n_components = n_components_range[np.argmin(bics)]

        print(f"Best number of components: {best_n_components}")

        if best_n_components == 1:
            predicted_course = [np.mean(courses)]
            probabilities_list.append(1.0)

        else: 
            # Fit a Gaussian Mixture Model with two components
            best_n_components = 2
            gmm = GaussianMixture(n_components=best_n_components).fit(courses)

            for i in range(gmm.n_components):
                print(f"Predicted course for direction {i}: {gmm.means_[i][0]:.2f} with probability {gmm.weights_[i]:.2f}")

            probabilities_list = gmm.weights_

            # Predict the course
            # predicted_course = gmm.predict(courses)
            predicted_course = gmm.means_
            predicted_course = [course[0] for course in predicted_course]

            # Calculate probabilities
            probabilities = gmm.predict_proba(courses)
            
    # print(f"Predicted courses: {predicted_course}")
    return predicted_course, average_distance, probabilities_list

def predict_next_point(course, average_distance, current_point):
    """Predict the next point based on average course and distance."""
    x2, y2 = current_point[2], current_point[3]
    x3 = x2 + average_distance * np.sin(np.radians(course))
    y3 = y2 + average_distance * np.cos(np.radians(course))
    print(f"x2: {x2:.2f}, y2: {y2:.2f}, x3: {x3:.2f}, y3: {y3:.2f}, course: {course:.2f}, avg_dist: {average_distance:.2f}")
    return [x2, y2, x3, y3]


class Path:
    def __init__(self, initial_points, probability):
        self.points = [initial_points]  # list of points; each point is a tuple (x, y)
        self.probability = probability  # probability of this path being chosen

    def add_point(self, point):
        self.points.append(point)

    def __repr__(self):
        return f"Path(prob={self.probability}, points={self.points})"

def check_point_within_bounds(point):
    """Check if the point is within the bounds of the map."""
    x, y = point[2], point[3]
    vertecis = [[-110, -110], [-80, -130], [-30, -115], [0, -120], [0, -90], [40, -60], [60, -50], [95, -20], [80, -10], [40, -8], [-20, -6], [-40, -25], [-52, -58], [-60, -68], [-110, -110]]
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
    return not inside

def iterative_path_prediction(initial_point, r_c, delta_course, K, X_B):
    paths = []  # This will store all possible paths as Path objects
    
    # Initial setup: create a starting path
    starting_path = Path(initial_point, 1.0)  # Start with a full probability
    current_points = [initial_point]  # Initialize current points
    paths.append(starting_path)

    stop_loop = False
    for k in range(K):
        if stop_loop:
            break
        print(f"Iteration {k+1}")
        new_current_points = []
        new_paths = []
        # print(f"Number of paths: {len(paths)}")
        # print(f"Current points: {current_points}")
        for path, current_point in zip(paths, current_points):
            # print(f"Current point: {current_point}")
            neighbours = find_closest_neighbours(X_B, current_point, r_c)
            if not neighbours:
                print("No neighbours found")
                new_paths.append(path)
                stop_loop = True
                break
            
            filtered_neighbours = filter_by_course(neighbours, current_point, delta_course)
            if not filtered_neighbours:
                print("No neighbours within course")
                new_paths.append(path)
                stop_loop = True
                break
            
            predicted_courses, avg_distance, probabilities = compute_probabilistic_course(filtered_neighbours)
            print(f"Predicted courses: {predicted_courses}")
            for course, prob in zip(predicted_courses, probabilities):
                # print(course, prob)
                # if prob < 0.35:
                #     print("Probability too low")
                #     new_paths.append(path)
                #     continue
                new_point = predict_next_point(course, avg_distance, current_point)
                print(f"Predicted point: {new_point}")
                # Need to check if the new point is within the bounds of the map
                if check_point_within_bounds(new_point):
                    print("Point outside bounds")
                    new_paths.append(path)
                    stop_loop = True
                    continue
                new_path = Path(new_point, path.probability*prob)  # New path probability
                # if new_path.probability < 0.45**k:
                #     continue
                new_path.points = path.points + [new_point]  # Copy old points and add new
                # print(new_path)
                new_paths.append(new_path)
                new_current_points.append(new_point)
                # print(f"New point: {new_point}")
        
        # Update paths and current points for the next iteration
        paths, current_points = new_paths, new_current_points
        print("\n")

        # # If more than two paths, keep only the two with highest probabilities
        max_num_paths = 8
        if len(paths) > max_num_paths:
            paths.sort(key=lambda p: p.probability, reverse=True)
            paths = paths[:max_num_paths]  # Keep top two probable paths
            # need to sort new_current_points as well
            current_points = [p.points[-1] for p in paths]
            # print(f"Current points: {current_points}")
            # will that work? or do we need to keep the old points as well?

    

        # If more than two paths, keep only the two with highest probabilities
        
    # print("\n")
    print(f"Number of paths: {len(paths)}") 
    if len(paths) > 2:
        paths.sort(key=lambda p: p.probability, reverse=True)
        paths = paths[:2]  # Keep top two probable paths

    # for path in paths:
    #     print(path)
    #     print("\n")
    return paths

def check_average_lenght_of_sub_tracks(X_B):
    sub_track_lengths = []
    # total_distance = 0
    count = 0
    for track_id, track_points in X_B.items():
        for sub_track in track_points:
            sub_track_temp_length = 0
            for i in range(0, len(sub_track)-2, 2):
                temp = calculate_distance(sub_track[i:i+2], sub_track[i+2:i+4])
                sub_track_temp_length += temp
            sub_track_lengths.append(sub_track_temp_length/2)

    # average_distance = total_distance / count
    # print(f"Average distance: {average_distance}")
    print(f"Average sub-track length: {np.mean(sub_track_lengths)}")

def main():
    # Load the data
    X_B = np.load('npy_files/X_B.npy', allow_pickle=True).item()

    # Define the initial point and angle
    initial_point = [65.7, -35.3]
    # initial_point = [65.7, -20]
    initial_point = [45, -20]
    initial_point = [35, -18]
    initial_point = [45, -20]
    initial_point = [-18, -15]
    angle = 180

    # initial_point = [-16, -111]
    # angle = 0

    # initial_point = [65.7, -35.3]
    # initial_point = [60, -20]
    # angle = - 110

    initial_point = find_initial_points(initial_point,angle)

    # Parameters for iterative path prediction
    r_c = 7
    delta_course = 20
    K = 23  # Total iterations
    K = 50

    # Run the iterative path prediction algorithm
    pred_paths = iterative_path_prediction(initial_point, r_c, delta_course, K, X_B)

    # Plot the predicted path
    ax, origin_x, origin_y, legend_elements = plot_all_vessel_tracks(X_B)
    plot_predicted_path_new(ax, pred_paths, origin_x, origin_y, legend_elements)
    plt.show()

if __name__ == '__main__':
    main()
