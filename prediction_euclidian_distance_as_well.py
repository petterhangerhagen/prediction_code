import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from plotting import start_plot, plot_all_vessel_tracks, plot_single_vessel_track, plot_predicted_path
from utilities import generate_random_point_and_angle_in_polygon, check_point_within_bounds, calculate_course
from utilities import calculate_distance, eucldian_distance, find_initial_points, predict_next_point

def find_closest_neighbours(X_B, point, radius):
    closest_neighbours = {}
    for track_id, track in X_B.items():
        for sub_track in track:
            euclidean_distance = eucldian_distance(point, sub_track[:4])
            if euclidean_distance <= radius:
                closest_neighbours.setdefault(track_id, []).append(sub_track)
    return closest_neighbours

def choice_of_number_of_components_old(courses):
    # List to store the BIC values for different numbers of components
    bics = []
    aics = []
    max_components = 8
    n_components_range = range(1, max_components +1)
    for n_components in n_components_range:
        # Fit a GMM with n_components
        gmm = GaussianMixture(n_components=n_components).fit(courses)
        # Append the BIC for this model to the list
        bics.append(gmm.bic(courses))
        aics.append(gmm.aic(courses))

    best_n_components_bic = n_components_range[np.argmin(bics)]
    best_n_components_aic = n_components_range[np.argmin(aics)]
   
    best_n_components = best_n_components_bic
    return best_n_components

def choice_of_number_of_components(data):
    # Compute BIC to determine the best number of components
    bics = []
    n_components_range = range(1, 9)  # Assuming up to 8 components
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components).fit(data)
        bics.append(gmm.bic(data))
    best_n = n_components_range[np.argmin(bics)]
    return best_n

def compute_probabilistic_course(neighbours):
    courses = []  # List to store all course values
    distances = []
    data = []
    total_distance = count =  0
    for tracks in neighbours.values():
        for track in tracks:
            course = calculate_course(track[2:4], track[4:6])
            courses.append(course)  # Add the course to the list
            distance = calculate_distance(track[2:4], track[4:6])
            distances.append(distance)
            data.append([course, distance])
            total_distance += distance
            count += 1
    average_distance = total_distance / count
    data = np.array(data)
    np.save("data.npy", data)
    temp_in = input("Press enter to continue")

    # if data.shape[0] < 60:
    #     # If not enough samples, use simple mean
    #     predicted = [np.mean(data, axis=0)]
    #     probabilities_list = [1.0]
    # else:
    best_n_components = choice_of_number_of_components(data)
    print(f"Best number of components: {best_n_components}")
    gmm = GaussianMixture(n_components=best_n_components).fit(data)
    probabilities_list = gmm.weights_
    predicted = gmm.means_  # This will include both course and distance
    print(f"Predicted: {predicted}")
    temp_in = input("Press enter to continue")
    return predicted, probabilities_list

    # # Reshape courses for GMM
    # courses = np.array(courses).reshape(-1, 1)
    # # print(f"Number of samples: {courses.shape[0]}")
    # probabilities_list = []
    # if courses.shape[0] < 60:
    #     predicted_course = [np.mean(courses)]
    #     probabilities_list.append(1.0)
    # else:
    #     best_n_components = choice_of_number_of_components(courses)
    #     print(f"Best number of components: {best_n_components}")
    #     if best_n_components == 1:
    #         predicted_course = [np.mean(courses)]
    #         probabilities_list.append(1.0)
    #     else: 
    #         # Fit a Gaussian Mixture Model with two components
    #         # best_n_components = 2
    #         gmm = GaussianMixture(n_components=best_n_components).fit(courses)
    #         probabilities_list = gmm.weights_
    #         predicted_course = gmm.means_
    #         predicted_course = [course[0] for course in predicted_course] # Rearange from 2D to 1D
            
    # return predicted_course, average_distance, probabilities_list

def calculate_similarity(course, predicted_courses):
    """Calculate the similarity between the current course and predicted courses."""
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

        neighbours = find_closest_neighbours(X_B, current_point, r_c)
        print(f"Number of closest neighbours: {len(neighbours)}")
        if not neighbours:
            break
    
        # predicted_courses, average_distance, probabilities_list = compute_probabilistic_course(neighbours)
        # print(f"Predicted courses: {predicted_courses}")
        # print(f"Probabilities: {probabilities_list}")

        predicted, probabilities_list = compute_probabilistic_course(neighbours)
        # print(f"Predicted courses: {predicted}")
        # print(f"Probabilities: {probabilities_list}")
        max_prob_index = np.argmax(probabilities_list)
        pred_course = predicted[max_prob_index][0]
        pred_distance = predicted[max_prob_index][1]
        print(f'Chosen Predicted Course: {pred_course:.2f}, Average Distance: {pred_distance:.2f}')
        # temp_in = input("Press enter to continue")

        # # Find the course with the highest probability
        # max_prob_index = np.argmax(probabilities_list)
        # pred_course = predicted_course[max_prob_index]

         # Calculate similarities with the current course
        predicted_courses = [pred[0] for pred in predicted]
        similarities = calculate_similarity(current_course, predicted_courses)
        print("\n")
        print(f"curent course: {current_course}")
        print(f"Predicted courses: {predicted_courses}")
        print(f"Similarities: {similarities}")
        
        # # Weight similarities by probabilities
        print(f"Probabilities: {probabilities_list}")
        weighted_scores = similarities * np.array(probabilities_list)
        print(f"Weighted scores: {weighted_scores}")
        # temp_in = input("Press enter to continue") 


        max_score_index = np.argmax(weighted_scores)
        pred_course = predicted[max_score_index][0]
        pred_distance = predicted[max_score_index][1]
        # pred_course = predicted_courses[max_score_index]
        # current_course = pred_course 

        # print(f'Chosen Predicted Course: {pred_course:.2f}, Average Distance: {average_distance:.2f}')
        

        # print(f'Predicted course: {pred_course:.2f}, Average distance: {avg_distance:.2f}')
        # current_point = predict_next_point(pred_course, average_distance, current_point)
        # if not check_point_within_bounds(current_point):
        #     print(f'Point outside bounds at iteration {k}.')
        #     break
        # print(f'Next point: {current_point}')
        # point_list.append(current_point[:2])
        # print("\n")

        current_point = predict_next_point(pred_course, pred_distance, current_point)
        if not check_point_within_bounds(current_point):
            print(f'Point outside bounds at iteration {k}.')
            break
        print(f'Next point: {current_point}')
        point_list.append(current_point[:2])
        print("\n")

    point_list.append(current_point[2:4])
    return point_list

def main():
    # Load the data
    X_B = np.load('npy_files/X_B.npy', allow_pickle=True).item()

    # Generate random point and angle in the given area
    area = "west"
    area = "east"
    area = "south"
    area = "north"
    for i in range(10):
        random_point, random_angle = generate_random_point_and_angle_in_polygon(area,X_B, plot=False)
        initial_point = find_initial_points(random_point,random_angle)

        # Parameters for iterative path prediction
        r_c = 7
        delta_course = 25
        K = 50 # Total iterations

        # Run the iterative path prediction algorithm
        pred_paths = iterative_path_prediction(initial_point, r_c, delta_course, K, X_B)

        # Plotting
        ax, origin_x, origin_y = start_plot()
        ax, origin_x, origin_y, legend_elements = plot_all_vessel_tracks(ax, X_B, origin_x, origin_y,save_plot=False)
        plot_predicted_path(ax, pred_paths, initial_point, random_angle, r_c, K, origin_x, origin_y,legend_elements,save_plot=True)
    # plt.show()

if __name__ == '__main__':
    main()
