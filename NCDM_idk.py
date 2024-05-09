import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from plotting import (
    start_plot, 
    plot_all_vessel_tracks, 
    plot_predicted_path, 
    plot_single_vessel_track, 
    plot_histogram, 
    plot_close_neigbors, 
    plot_histogram_2,
    plot_recursive_paths
)
from utilities import (
    generate_random_point_and_angle_in_polygon,
    check_point_within_bounds,
    calculate_course,
    calculate_distance,
    eucldian_distance,
    find_initial_points,
    predict_next_point,
    add_CVM
)
from check_start_and_stop import CountMatrix
from GMM_components import get_GMM, get_GMM_modified


class NCDM:
    def __init__(self, data_file):
        self.X_B = np.load(data_file, allow_pickle=True).item()
        self.num_tracks = len(self.X_B)
        self.gmm_components = 8
        self.gmm_margin = 60
        self.CVM = False
        self.compare_to_track = False
        self.plot_histogram = False

        self.r_c = 10
        self.K = 100
        self.track_id = None
        with open(f"test.txt", "w") as f:
            f.write("")

        self.point_dict = {}
        self.recursive_counter = 0

    def find_track(self, track_id):
        self.track_id = track_id
        self.track = self.X_B[track_id]
        track_initial_point = self.track[0][:4]
        self.compare_to_track = True
        return track_initial_point

    def find_closest_neighbours(self, point, radius):
        closest_neighbours = {}
        for track_id, track in self.X_B.items():
            for sub_track in track:
                euclidean_distance = eucldian_distance(point, sub_track[:4])
                if euclidean_distance <= radius:
                    closest_neighbours.setdefault(track_id, []).append(sub_track)
        if self.CVM:
            closest_neighbours = add_CVM(closest_neighbours, point)
        return closest_neighbours

    def compute_probabilistic_course(self, neighbours):
        courses = []
        distances = []
        total_distance = 0
        count = 0
        for key,values in neighbours.items():
            for sub_track in values:
                course = calculate_course(sub_track[2:4], sub_track[4:6])
                courses.append(course)
                _distance = calculate_distance(sub_track[2:4], sub_track[4:6])
                distances.append(_distance)
                total_distance += _distance
                count += 1
        average_distance = total_distance / count
        courses = np.array(courses).reshape(-1, 1)
        data = np.array(courses)
        if len(data) < 2:
            predicted_course = data[0]
            probabilities_list = [1.0]
        else:
            gmm = get_GMM_modified(data, max_comps=self.gmm_components, margin=self.gmm_margin)
            probabilities_list = gmm.weights_
            predicted = gmm.means_
            predicted_course = [course[0] for course in predicted]
        return data,gmm, predicted_course, average_distance, probabilities_list
        

    def calculate_similarity(self, course, predicted_courses):
        similarities = np.abs(predicted_courses - course) % 360
        similarities = np.minimum(similarities, 360 - similarities)
        normalized_similarities = 1 - (similarities / 180.0)
        return normalized_similarities
    
    def calculate_similarity_new(self, course, predicted_courses):
        similarities = np.abs(predicted_courses - course)
        modified_similarities = np.zeros(len(similarities))
        for i in range(len(similarities)):
            if similarities[i] < 5:
                modified_similarities[i] = 1
            elif similarities[i] < 10:
                modified_similarities[i] = 0.8
            elif similarities[i] < 20:
                modified_similarities[i] = 0.6
            elif similarities[i] < 30:
                modified_similarities[i] = 0.4
            else:
                modified_similarities[i] = 0
        return modified_similarities

    def iterative_path_prediction(self, initial_point, r_c, K, recursive_run=False):
        self.recursive_counter += 1
        point_list = [initial_point[:2]]
        current_point = initial_point
        current_course = calculate_course(current_point[:2], current_point[2:4])
        for k in range(K):
            print(f"Iteration {k}")
            print(f"Current point: {current_point}")
            current_course = calculate_course(current_point[:2], current_point[2:4])
            neighbours = self.find_closest_neighbours(current_point, r_c)
            if not neighbours:
                break
            data, gmm, predicted_courses, average_distance, probabilities_list = self.compute_probabilistic_course(neighbours)

            # Check the predicted courses if they are within the bounds
            for course in predicted_courses:
                if np.abs(course - current_course) > 100:
                    probabilities_list = np.delete(probabilities_list, predicted_courses.index(course))
                    predicted_courses.remove(course)
            
            
                

            max_prob_index = np.argmax(probabilities_list)
            pred_course = predicted_courses[max_prob_index]
            current_course = pred_course

            if not recursive_run:
                if len(predicted_courses) > 1:
                    # Want to save the others not used
                    predicted_courses_copy = predicted_courses.copy()
                    probabilities_list_copy = probabilities_list.copy()
                    predicted_courses_copy.pop(max_prob_index)
                    probabilities_list_copy = np.delete(probabilities_list_copy, max_prob_index)

                    for i in range(len(probabilities_list_copy)):
                        if probabilities_list_copy[i] > 0.1:
                            new_pred_point = predict_next_point(predicted_courses_copy[i], average_distance, current_point)
                            new_point_list = self.iterative_path_prediction(new_pred_point, r_c, K, recursive_run=True)
                            self.point_dict[f"Point {self.recursive_counter}_{i}"] = point_list + new_point_list

                            with open(f"test.txt", "a") as f:
                                f.write(f"Iteration {k}, predicted_courses = {predicted_courses_copy[i]}, predicted_probs = {probabilities_list_copy[i]} \n")
                                temp_pred_point = predict_next_point(predicted_courses_copy[i], average_distance, current_point)
                                f.write(f"Predicted point = {temp_pred_point} \n")
                                # Do the list to str
                                str_point_list = str(point_list)
                                f.write(str_point_list)
                                f.write("\n")
                            

                        

            current_point = predict_next_point(pred_course, average_distance, current_point)
            if not check_point_within_bounds(current_point):
                break
            point_list.append(current_point[:2])

            if self.plot_histogram:
                plot_histogram_2(data, gmm, point_list, self.track_id, predicted_courses, probabilities_list, save_plot=True)

            print("\n")
        point_list.append(current_point[2:4])
        return point_list

    def run_prediction(self, initial_point):
        pred_paths = self.iterative_path_prediction(initial_point, self.r_c, self.K)
        ax, origin_x, origin_y = start_plot()
        ax, origin_x, origin_y, legend_elements = plot_all_vessel_tracks(ax, self.X_B, origin_x, origin_y,
                                                                         save_plot=False)
        ax, origin_x, origin_y, legend_elements = plot_predicted_path(ax, pred_paths, initial_point, self.r_c, self.K, origin_x, origin_y,
                            legend_elements, save_plot=False)
        if self.compare_to_track:
            plot_single_vessel_track(ax, self.track, origin_x, origin_y, legend_elements, self.track_id, save_plot=True)

        np.save(f"npy_files/predicted_paths.npy", self.point_dict)
        # for key, value in self.point_dict.items():
        #     ax, origin_x, origin_y, legend_elements = plot_predicted_path(ax, value, initial_point, self.r_c, self.K, origin_x, origin_y,
        #                     legend_elements, save_plot=False, color='r')

if __name__ == '__main__':
    path_predictor = NCDM('npy_files/X_B.npy')
    print(path_predictor.num_tracks)

    # for i in range(path_predictor.num_tracks):
    #     initial_point = path_predictor.find_track(i)
    #     path_predictor.run_prediction(initial_point)

    initial_point = path_predictor.find_track(0)        # 5
    path_predictor.run_prediction(initial_point)

    # # initial_point = [-23, -105]
    # # random_angle = -10
    # # initial_point = find_initial_points(initial_point, random_angle)

    filename = "npy_files/predicted_paths.npy"
    plot_recursive_paths(filename)
