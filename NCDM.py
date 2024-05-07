import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from plotting import start_plot, plot_all_vessel_tracks, plot_predicted_path
from utilities import (
    generate_random_point_and_angle_in_polygon,
    check_point_within_bounds,
    calculate_course,
    calculate_distance,
    eucldian_distance,
    find_initial_points,
    predict_next_point,
)
from check_start_and_stop import CountMatrix
from GMM_components import get_GMM


class NCDM:
    def __init__(self, data_file):
        self.X_B = np.load(data_file, allow_pickle=True).item()
        self.gmm_components = 5
        self.gmm_margin = 30
        self.CVM = False

    def constant_velocity_model(self, point):
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

    def add_CVM(self, closest_neighbours, point):
        alpha = 100 
        M = len(closest_neighbours)
        if M == 0:
            W = 1
        else:
            W = round(alpha / M)
        for i in range(W):
            closest_neighbours.setdefault(-1, []).append(self.constant_velocity_model(point))

    def find_closest_neighbours(self, point, radius):
        closest_neighbours = {}
        for track_id, track in self.X_B.items():
            for sub_track in track:
                euclidean_distance = eucldian_distance(point, sub_track[:4])
                if euclidean_distance <= radius:
                    closest_neighbours.setdefault(track_id, []).append(sub_track)
        if self.CVM:
            self.add_CVM(closest_neighbours, point)

        return closest_neighbours

    def compute_probabilistic_course(self, neighbours):
        courses = []
        total_distance = count = 0
        for tracks in neighbours.values():
            for track in tracks:
                course = calculate_course(track[2:4], track[4:6])
                courses.append(course)
                total_distance += calculate_distance(track[2:4], track[4:6])
                count += 1
        average_distance = total_distance / count
        courses = np.array(courses).reshape(-1, 1)
        data = np.array(courses)
        if len(data) < 2:
            predicted_course = data[0]
            probabilities_list = [1.0]
        else:
            gmm = get_GMM(data, max_comps=self.gmm_components, margin=self.gmm_margin)
            probabilities_list = gmm.weights_
            predicted = gmm.means_
            predicted_course = [course[0] for course in predicted]
        return predicted_course, average_distance, probabilities_list

    def calculate_similarity(self, course, predicted_courses):
        similarities = np.abs(predicted_courses - course) % 360
        similarities = np.minimum(similarities, 360 - similarities)
        normalized_similarities = 1 - (similarities / 180.0)
        return normalized_similarities

    def iterative_path_prediction(self, initial_point, r_c, K):
        point_list = [initial_point[:2]]
        current_point = initial_point
        current_course = calculate_course(current_point[:2], current_point[2:4])
        for k in range(K):
            print(f"Iteration {k}")
            current_course = calculate_course(current_point[:2], current_point[2:4])
            neighbours = self.find_closest_neighbours(current_point, r_c)
            if not neighbours:
                break
            predicted_courses, average_distance, probabilities_list = self.compute_probabilistic_course(neighbours)

            choice = 1
            if choice == 0:
                max_prob_index = np.argmax(probabilities_list)
                pred_course = predicted_courses[max_prob_index]
                current_course = pred_course
            elif choice == 1:
                similarities = self.calculate_similarity(current_course, predicted_courses)
                weighted_scores = similarities * np.array(probabilities_list)
                max_score_index = np.argmax(weighted_scores)
                pred_course = predicted_courses[max_score_index]
                current_course = pred_course
            current_point = predict_next_point(pred_course, average_distance, current_point)
            if not check_point_within_bounds(current_point):
                break
            point_list.append(current_point[:2])
        point_list.append(current_point[2:4])
        return point_list

    def run_prediction(self, initial_point, r_c, K):
        pred_paths = self.iterative_path_prediction(initial_point, r_c, K)
        ax, origin_x, origin_y = start_plot()
        ax, origin_x, origin_y, legend_elements = plot_all_vessel_tracks(ax, self.X_B, origin_x, origin_y,
                                                                         save_plot=False)
        plot_predicted_path(ax, pred_paths, initial_point, random_angle, r_c, K, origin_x, origin_y,
                            legend_elements, save_plot=True)


if __name__ == '__main__':
    path_predictor = NCDM('npy_files/X_B.npy')
    initial_point = [-23, -105]
    random_angle = -10
    initial_point = find_initial_points(initial_point, random_angle)
    path_predictor.run_prediction(initial_point, r_c=6, K=100)
