import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from plotting import start_plot, plot_all_vessel_tracks, plot_single_vessel_track, plot_predicted_path
from utilities import generate_random_point_and_angle_in_polygon, check_point_within_bounds, calculate_course
from utilities import calculate_distance, eucldian_distance, find_initial_points, predict_next_point
from check_start_and_stop import CountMatrix
from GMM_components import get_GMM

def constant_velocity_model(point):
    T = 1
    sigma_a = 0.1
    Q = sigma_a**2 * np.array([[(T**4)/3, (T**3)/2, 0, 0],[(T**3)/2, T**2, 0, 0],[0, 0, (T**4)/3, (T**3)/2],[0, 0, (T**3)/2, T**2]])
    F = np.array([[1, T, 0, 0],[0, 1, 0, 0],[0, 0, 1, T],[0, 0, 0, 1]])
    t = 1.0
    current_state = np.array([point[2], (1/t)*(point[2]-point[0]), point[3], (1/t)*(point[3]-point[1])])
    predicted_state = np.dot(F, current_state) + np.random.multivariate_normal([0, 0, 0, 0], Q)
    sub_trajectory = [point[0], point[1], point[2], point[3], predicted_state[0], predicted_state[2]]
    return sub_trajectory

def find_closest_neighbours(X_B, point, radius):
    closest_neighbours = {}
    for track_id, track in X_B.items():
        for sub_track in track:
            euclidean_distance = eucldian_distance(point, sub_track[:4])
            if euclidean_distance <= radius:
                closest_neighbours.setdefault(track_id, []).append(sub_track)

    # alpha = 100 
    # M = len(closest_neighbours)
    # if M == 0:
    #     W = 1
    # else:
    #     W = round(alpha / M)
    # print(f"Adding {W} constant velocity points to the closest neighbours.")

    # for i in range(W):
    #     closest_neighbours.setdefault(-1, []).append(constant_velocity_model(point))

    return closest_neighbours


def compute_probabilistic_course(neighbours):
    courses = []  # List to store all course values
    total_distance = count =  0
    for tracks in neighbours.values():
        for track in tracks:
            course = calculate_course(track[2:4], track[4:6])
            courses.append(course)  # Add the course to the list
            total_distance += calculate_distance(track[2:4], track[4:6])
            count += 1
    average_distance = total_distance / count
            
    # Reshape courses for GMM
    courses = np.array(courses).reshape(-1, 1)
    data = np.array(courses)
    if len(data) < 2:
        predicted_course = data[0]
        probabilities_list = [1.0]
        return predicted_course, average_distance, probabilities_list
    
    # Fit GMM to the data
    # gmm = get_GMM(data, max_comps=5, margin=10)
    gmm = get_GMM(data, max_comps=5, margin=30)
    probabilities_list = gmm.weights_
    predicted = gmm.means_  # This will include both course and distance

    predicted_course = [course[0] for course in predicted] # Rearange from 2D to 1D
    return predicted_course, average_distance, probabilities_list

def calculate_similarity(course, predicted_courses):
    """Calculate the similarity between the current course and predicted courses."""
    # If the course is larger than 360 need to use modulo
    similarities = np.abs(predicted_courses - course) % 360
    similarities = np.minimum(similarities, 360 - similarities)  # Adjust for circular nature of angles
    # Normalize similarities to a range between 0 and 1
    normalized_similarities = 1 - (similarities / 180.0)
    return normalized_similarities

def iterative_path_prediction(initial_point, r_c, delta_course, K, X_B):
    """Iteratively predict path based on initial point and movement statistics."""

    point_list = [initial_point[:2]]
    current_point = initial_point
    current_course = calculate_course(current_point[:2], current_point[2:4])

    for k in range(K):
        print(f'Iteration {k}')
        current_course = calculate_course(current_point[:2], current_point[2:4])
        print(f"Current course: {current_course:.2f}")

        neighbours = find_closest_neighbours(X_B, current_point, r_c)
        print(f"Number of closest neighbours: {len(neighbours)}")
        if not neighbours:
            break

        # Filter neighbours by course
        # neighbours = filter_by_course(neighbours, current_course, delta_course)
        # if not neighbours:
        #     print(f'No neighbours found within course range at iteration {k}.')
        #     break

        predicted_courses, average_distance, probabilities_list = compute_probabilistic_course(neighbours)
        print(f"Predicted courses: {predicted_courses}")
        
        choice = 1
        if choice == 0:
            # Find the course with the highest probability
            max_prob_index = np.argmax(probabilities_list)
            pred_course = predicted_courses[max_prob_index]
            current_course = pred_course
            # print(f"Probabilities: {probabilities_list}")
            # print(f'Predicted Course: {pred_course:.2f}, Average Distance: {average_distance:.2f}')
            
        elif choice == 1:
            # Calculate similarities with the current course
            similarities = calculate_similarity(current_course, predicted_courses)
            # Weight similarities by probabilities
            weighted_scores = similarities * np.array(probabilities_list)
            max_score_index = np.argmax(weighted_scores)
            pred_course = predicted_courses[max_score_index]
            current_course = pred_course 
            # print(f'Weighted scores: {weighted_scores}')
            # print(f'Chosen Predicted Course: {pred_course:.2f}, Average Distance: {average_distance:.2f}')
    
        current_point = predict_next_point(pred_course, average_distance, current_point)
        if not check_point_within_bounds(current_point):
            print(f'Point outside bounds at iteration {k}.')
            break
        # print(f'Next point: {current_point}')
        point_list.append(current_point[:2])
        print("\n")

    point_list.append(current_point[2:4])
    return point_list

if __name__ == '__main__':
    # Load the data
    X_B = np.load('npy_files/X_B.npy', allow_pickle=True).item()

    # ax, origin_x, origin_y = start_plot()
    # ax, origin_x, origin_y, legend_elements = plot_all_vessel_tracks(ax, X_B, origin_x, origin_y,save_plot=True)

    # count_matrix = CountMatrix(reset=True)

    # Generate random point and angle in the given area
    # area = "F"
    for i in range(1):
        # random_point, random_angle = generate_random_point_and_angle_in_polygon(area, X_B, plot=False)
        # initial_point = find_initial_points(random_point,random_angle)

        # initial_point = [40, -20.6]
        # random_angle = -90
        # initial_point = find_initial_points(initial_point,random_angle)

        initial_point = [-23, -105]
        random_angle = -10
        initial_point = find_initial_points(initial_point,random_angle)

        
        # Parameters for iterative path prediction
        r_c = 6
        delta_course = 60
        K = 100 # Total iterations

        # Run the iterative path prediction algorithm
        pred_paths = iterative_path_prediction(initial_point, r_c, delta_course, K, X_B)

        # Plotting
        ax, origin_x, origin_y = start_plot()
        ax, origin_x, origin_y, legend_elements = plot_all_vessel_tracks(ax, X_B, origin_x, origin_y,save_plot=False)
        plot_predicted_path(ax, pred_paths, initial_point, random_angle, r_c, K, origin_x, origin_y,legend_elements,save_plot=True)
        # plt.show()
        # count_matrix.check_stop(pred_paths[0],pred_paths[-1],area)

    # print(count_matrix.unvalidated_track)