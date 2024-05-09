import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from plotting import start_plot, plot_all_vessel_tracks, plot_predicted_path, plot_single_vessel_track, plot_histogram, plot_close_neigbors
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

    def iterative_path_prediction(self, initial_point, r_c, K):
        self.point_list = [initial_point[:2]]
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

            choice = 1
            if choice == 0:
                max_prob_index = np.argmax(probabilities_list)
                pred_course = predicted_courses[max_prob_index]
                current_course = pred_course
            elif choice == 1:
                # print(f"Current course: {current_course:.2f}")
                courses_to_remove = []
                for pred_course in predicted_courses:
                    # print("---------------------------")
                    # print(f"Predicted course: {pred_course:.2f}")
                    # print(f"{np.abs(pred_course - current_course):.2f}")
                    if (np.abs(pred_course - current_course) > 70):
                        # print(f"Removing course: {pred_course:.2f}")
                        courses_to_remove.append(pred_course)
                        # print(f"{np.abs(pred_course - current_course):.2f}")
                        # probabilities_list = np.delete(probabilities_list, predicted_courses.index(pred_course))
                        # predicted_courses.remove(pred_course)
                for course in courses_to_remove:
                    probabilities_list = np.delete(probabilities_list, predicted_courses.index(course))
                    predicted_courses.remove(course)
                # print(f"Remaining courses: {predicted_courses}")
                # print("#########################")
                max_prob_index = np.argmax(probabilities_list)
                pred_course = predicted_courses[max_prob_index]
                # print(f"Predicted course: {pred_course}")
                # print(f"Current course: {current_course}")
                current_course = pred_course
            elif choice == 2:
                similarities = self.calculate_similarity(current_course, predicted_courses)
                weighted_scores = similarities * np.array(probabilities_list)
                max_score_index = np.argmax(weighted_scores)
                pred_course = predicted_courses[max_score_index]
                current_course = pred_course

            current_point = predict_next_point(pred_course, average_distance, current_point)
            if not check_point_within_bounds(current_point):
                break
            self.point_list.append(current_point[:2])

            if self.plot_histogram:
                if choice == 0 or choice == 1:
                    plot_histogram(data, gmm, self.point_list, self.track_id, save_plot=True)
                elif choice == 1:
                    plot_histogram(data, gmm, self.point_list, self.track_id, sim=similarities, weight=weighted_scores, save_plot=True)

            print("\n")
        self.point_list.append(current_point[2:4])
        return self.point_list

    def run_prediction(self, initial_point):
        pred_paths = self.iterative_path_prediction(initial_point, self.r_c, self.K)
        ax, origin_x, origin_y = start_plot()
        ax, origin_x, origin_y, legend_elements = plot_all_vessel_tracks(ax, self.X_B, origin_x, origin_y,
                                                                         save_plot=False)
        ax, origin_x, origin_y, legend_elements = plot_predicted_path(ax, pred_paths, initial_point, self.r_c, self.K, origin_x, origin_y,
                            legend_elements, save_plot=False)
        if self.compare_to_track:
            plot_single_vessel_track(ax, self.track, origin_x, origin_y, legend_elements, self.track_id, save_plot=True)

    def calculate_root_mean_square_error(self):
        # Can only be called after run_prediction
        try:
            pred_path = self.point_list
            actual_path = self.track
        except:
            print("No path found")
            return None
        error = 0
        # pred_path = np.load("npy_files/points_list.npy")
        # actual_path = np.load("npy_files/track.npy")
        actual_path = [point[:2] for point in actual_path]
        

        if len(pred_path) != len(actual_path):
            # choose the shortest path
            if len(pred_path) > len(actual_path):
                pred_path = pred_path[:len(actual_path)]
            else:
                actual_path = actual_path[:len(pred_path)]

        error = 0
        for i in range(len(pred_path)):
            # print(f"pred path {pred_path[i]}")
            # print(f"actual path {actual_path[i]}")
            error += (pred_path[i][0] - actual_path[i][0])**2 + (pred_path[i][1] - actual_path[i][1])**2
            # print("\n")
        error = np.sqrt(error / len(pred_path))
        print(f"Root mean square error: {error}")

        fig, ax = plt.subplots()
        ax.plot([point[0] for point in pred_path], [point[1] for point in pred_path],"-o", label="Predicted path")
        ax.plot([point[0] for point in actual_path], [point[1] for point in actual_path],"-o", label="Actual path")
        ax.legend()
        plt.show()
        # return error

if __name__ == '__main__':
    path_predictor = NCDM('npy_files/X_B.npy')
    print(path_predictor.num_tracks)
    # # for i in range(path_predictor.num_tracks):
    # #     initial_point = path_predictor.find_track(i)
    # #     path_predictor.run_prediction(initial_point)

    initial_point = path_predictor.find_track(6)        # 5
    path_predictor.run_prediction(initial_point)
    path_predictor.calculate_root_mean_square_error()
    # np.save("npy_files/points_list.npy", path_predictor.point_list)
    # np.save("npy_files/track.npy", path_predictor.track)
    # # # initial_point = [-23, -105]
    # # # random_angle = -10
    # # # initial_point = find_initial_points(initial_point, random_angle)
    
