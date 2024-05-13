import numpy as np
import matplotlib.pyplot as plt
import sys
from plotting import (
    start_plot, 
    plot_all_vessel_tracks, 
    plot_predicted_path, 
    plot_single_vessel_track, 
    plot_histogram, 
    plot_close_neigbors,
    plot_histogram_distances,
    plot_histogram_courses,
    angle_constraints_demo
)
from utilities import (
    generate_random_point_and_angle_in_polygon,
    check_point_within_bounds,
    calculate_course,
    calculate_distance,
    eucldian_distance,
    find_initial_points,
    predict_next_point,
    add_CVM,
    RMSE,
    read_results,
    find_best_rc
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
        self.course_diff = 90
        self.choice = 1

    def find_track(self, track_id):
        self.track_id = track_id
        self.track = self.X_B[track_id]
        track_initial_point = self.track[0][:4]
        self.track = [track[:2] for track in self.track]
        self.compare_to_track = True
        return track_initial_point

    def find_closest_neighbours(self, point, radius):
        closest_neighbours = {}
        count = 0
        for track_id, track in self.X_B.items():
            for sub_track in track:
                euclidean_distance = eucldian_distance(point, sub_track[:4])
                if euclidean_distance <= radius:
                    closest_neighbours.setdefault(track_id, []).append(sub_track)
                    count += 1
        if self.CVM:
            closest_neighbours = add_CVM(closest_neighbours, point)

        print(f"Number of neighbours: {count}")
        return closest_neighbours

    def compute_average_course_and_distance(self, neighbours):
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
        average_course = np.mean(courses)
        return average_course, average_distance

    def compute_probabilistic_course(self, neighbours, iteration_num=0):
        courses = []
        distances = []
        total_distance = 0
        count = 0
        data2 = []
        for key,values in neighbours.items():
            for sub_track in values:
                course = calculate_course(sub_track[2:4], sub_track[4:6])
                courses.append(course)
                _distance = calculate_distance(sub_track[2:4], sub_track[4:6])
                distances.append(_distance)
                data2.append([course, _distance])
                total_distance += _distance
                count += 1
        
        # # Demonstation of distance distribution
        # plot_histogram_distances(distances, self.point_list, self.X_B, track_id=self.track_id, iteration_num=iteration_num, save_plot=True)
        
        average_distance = total_distance / count
        courses = np.array(courses).reshape(-1, 1)
        temp_course_1 = np.array(courses) % 360
        temp_course_2 = temp_course_1 - 360
        course_complete = np.concatenate((temp_course_1, temp_course_2))
        data = np.array(course_complete)
        if len(data) < 2:
            predicted_course = data[0]
            probabilities_list = [1.0]
        else:
            gmm = get_GMM_modified(data, max_comps=self.gmm_components, margin=self.gmm_margin)
            probabilities_list = gmm.weights_
            predicted = gmm.means_
            predicted_course = [course[0] for course in predicted]

            # # Demonstation of modifed course distribution
            # plot_histogram_courses(courses, gmm.n_components, self.point_list, self.X_B, track_id=self.track_id, iteration_num=iteration_num, save_plot=True)
            # plot_histogram_courses(data, gmm.n_components, self.point_list, self.X_B, track_id=self.track_id, iteration_num=iteration_num, save_plot=True)
            
            # Demonstation of angle constraints
            # angle_constraints_demo(data, gmm.n_components, self.gmm_components, self.gmm_margin, self.point_list, self.X_B, self.track, track_id=self.track_id, iteration_num=iteration_num, save_plot=False)


        return data,gmm, predicted_course, average_distance, probabilities_list
    
    def remove_courses_outside_range(self, courses, current_course, probabilities_list):
        normalized_course = (np.array(courses) + 180) % 360 - 180
        normalized_course = list(normalized_course)
        new_courses = []
        new_probabilities = []
        for course in normalized_course:
            diff = abs(course - current_course)
            if diff < self.course_diff:
                new_courses.append(course)
                new_probabilities.append(probabilities_list[normalized_course.index(course)])
        if not new_courses:
            new_courses = normalized_course
            new_probabilities = probabilities_list

        return new_courses, new_probabilities

    def iterative_path_prediction(self, initial_point, r_c, K):
        self.point_list = [initial_point[:2]]
        current_point = initial_point
        current_course = calculate_course(current_point[:2], current_point[2:4])
        for k in range(K):
            print(f"Iteration {k}, track id: {self.track_id} of {self.num_tracks}")
            print(f"Current point: {current_point[0]:.2f}, {current_point[1]:.2f}")
            current_course = calculate_course(current_point[:2], current_point[2:4])
            neighbours = self.find_closest_neighbours(current_point, r_c)
            if not neighbours:
                break

            data, gmm, predicted_courses, average_distance, probabilities_list = self.compute_probabilistic_course(neighbours, iteration_num=k)

            # self.choice = 0
            if self.choice == 0:
                max_prob_index = np.argmax(probabilities_list)
                pred_course = predicted_courses[max_prob_index]
                current_course = pred_course
            elif self.choice == 1:
                predicted_courses_mod, probabilities_list_mod = self.remove_courses_outside_range(predicted_courses, current_course, probabilities_list)
                max_prob_index = np.argmax(probabilities_list_mod)
                pred_course = predicted_courses_mod[max_prob_index]
                current_course = pred_course
            
            # pred_course, average_distance = self.compute_average_course_and_distance(neighbours)

            if average_distance < 2:
                print("Average distance too low")
                break
            
            print(f"Predicted course: {pred_course:.2f}")
            current_point = predict_next_point(pred_course, average_distance, current_point)
            if not check_point_within_bounds(current_point):
                break
            self.point_list.append(current_point[:2])

            if self.plot_histogram:
                plot_histogram(data, gmm, self.point_list, self.track_id, save_plot=True)
                
            print("\n")
        self.point_list.append(current_point[2:4])
        return self.point_list

    def run_prediction(self, initial_point):
        pred_paths = self.iterative_path_prediction(initial_point, self.r_c, self.K)
        self.calculate_root_mean_square_error()

        ax, origin_x, origin_y = start_plot()
        ax, origin_x, origin_y, legend_elements = plot_all_vessel_tracks(ax, self.X_B, origin_x, origin_y,
                                                                         save_plot=False)
        ax, origin_x, origin_y, legend_elements = plot_predicted_path(ax, pred_paths, initial_point, self.r_c, self.K, self.rmse, origin_x, origin_y,
                            legend_elements, save_plot=False)
        if self.compare_to_track:
            plot_single_vessel_track(ax, self.track, origin_x, origin_y, legend_elements, self.track_id, save_plot=False)

    def calculate_root_mean_square_error(self):
        # Can only be called after run_prediction
        try:
            pred_path = self.point_list
            actual_path = self.track
        except:
            print("No path found")
            return None
        error = 0
        self.rmse = RMSE(actual_path, pred_path, plot_statement=False)
        print(f"Root mean square error: {self.rmse}")


def main():
    if len(sys.argv) < 2:
        print("Please provide a track id")
        track_id = 0
    else:
        track_id = int(sys.argv[1])
        
    path_predictor = NCDM('npy_files/X_B.npy')
    path_predictor.r_c = 10
    path_predictor.choice = 1
    path_predictor.plot_histogram = False
    initial_point = path_predictor.find_track(track_id)
    path_predictor.run_prediction(initial_point)
    plt.show()

 
def main2():
    data = np.load('npy_files/X_B_filtered.npy', allow_pickle=True).item()
    keys = list(data.keys())
    path_predictor = NCDM('npy_files/X_B_filtered.npy')
    num_of_tracks = path_predictor.num_tracks

    # count_matrix = CountMatrix(reset=True)
    # tracks_to_check = [0,3,5,16,17,18,19,20,22,24,29,30,33,35,37,38,39,40,44,50,51,57,58,59]
    # for i in range(20,40):
    with open('results.txt', 'w') as f:
        f.write("")
    for i in keys:
        initial_point = path_predictor.find_track(i)
        path_predictor.r_c = 3
        path_predictor.choice = 0
        path_predictor.run_prediction(initial_point)
        rmse1 = path_predictor.rmse

        path_predictor.r_c = 10
        path_predictor.choice = 1
        path_predictor.run_prediction(initial_point)
        rmse2 = path_predictor.rmse
        with open('results.txt', 'a') as f:
            f.write(f"Track {i}: r_c =3, RMSE: {rmse1:.2f}, r_c=10 RMSE: {rmse2:.2f}\n")
        # count_matrix.check_start_and_stop_prediction(path_predictor.point_list[0], path_predictor.point_list[-1])
        plt.close('all')

if __name__ == '__main__':
    find_best_rc()
