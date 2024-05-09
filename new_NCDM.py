import numpy as np
import matplotlib.pyplot as plt
import heapq
from sklearn.mixture import GaussianMixture
from plotting import start_plot, plot_all_vessel_tracks, plot_predicted_path, plot_single_vessel_track, plot_histogram
from utilities import (
    generate_random_point_and_angle_in_polygon,
    check_point_within_bounds,
    calculate_course,
    calculate_distance,
    eucldian_distance,
    find_initial_points,
    predict_next_point,
    add_CVM,
    check_similarites_between_tracks,
    check_similarites_between_tracks_2
)
from check_start_and_stop import CountMatrix
from GMM_components import get_GMM, choice_of_number_of_components

class NCDM:
    def __init__(self, data_file):
        self.X_B = np.load(data_file, allow_pickle=True).item()
        self.num_tracks = len(self.X_B)
        self.gmm_components = 8
        self.gmm_margin = 60
        self.CVM = False
        self.compare_to_track = False
        self.plot_histogram = True

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
        for tracks in neighbours.values():
            for track in tracks:
                course = calculate_course(track[2:4], track[4:6])
                courses.append(course)
                _distance = calculate_distance(track[2:4], track[4:6])
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
            # best_n = choice_of_number_of_components(data)
            # print(f"Best number of components: {best_n}")
            # gmm = GaussianMixture(n_components=best_n).fit(data)
            gmm = get_GMM(data, max_comps=self.gmm_components, margin=self.gmm_margin)
            # print(f"Number of components: {gmm.n_components}")
            # if self.plot_histogram:
            #     plot_histogram(data, gmm, self.point_list, self.track_id, save_plot=True)
            probabilities_list = gmm.weights_
            predicted = gmm.means_
            predicted_course = [course[0] for course in predicted]
        return predicted_course, average_distance, probabilities_list

    def calculate_similarity_old(self, course, predicted_courses):
        similarities = np.abs(predicted_courses - course) % 360
        similarities = np.minimum(similarities, 360 - similarities)
        normalized_similarities = 1 - (similarities / 180.0)
        return normalized_similarities
    
    def calculate_similarity(self, course, predicted_courses):
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
            self.point_list.append(current_point[:2])
            print("\n")
        self.point_list.append(current_point[2:4])
        return self.point_list

    def iterative_path_prediction_multiple_predictions(self, initial_point, r_c, K):
        # self.point_list = [initial_point[:2]]
        prediction_paths = Predictions(initial_point[:2])
        current_point = initial_point
        current_course = calculate_course(current_point[:2], current_point[2:4])
        for k in range(K):
            print(f"Iteration {k}")
            current_course = calculate_course(current_point[:2], current_point[2:4])
            neighbours = self.find_closest_neighbours(current_point, r_c)
            if not neighbours:
                break
            predicted_courses, average_distance, probabilities_list = self.compute_probabilistic_course(neighbours)

            if predicted_courses > prediction_paths.max_number_of_paths:
                # need to find the most probable paths
                most_probable_index =np.argsort(probabilities_list)[-3:][::-1]
                for index in most_probable_index:
                    pred_course = predicted_courses[index]
                    current_point = predict_next_point(pred_course, average_distance, current_point)
                    if not check_point_within_bounds(current_point):
                        break
                    prediction_paths.add_path(current_point[:2], probabilities_list[index])
                    

        #     choice = 1
        #     if choice == 0:
        #         max_prob_index = np.argmax(probabilities_list)
        #         pred_course = predicted_courses[max_prob_index]
        #         current_course = pred_course
        #     elif choice == 1:
        #         similarities = self.calculate_similarity(current_course, predicted_courses)
        #         weighted_scores = similarities * np.array(probabilities_list)
        #         max_score_index = np.argmax(weighted_scores)
        #         pred_course = predicted_courses[max_score_index]
        #         current_course = pred_course
        #     current_point = predict_next_point(pred_course, average_distance, current_point)
        #     if not check_point_within_bounds(current_point):
        #         break
        #     self.point_list.append(current_point[:2])
        #     print("\n")
        # self.point_list.append(current_point[2:4])
        # return self.point_list

    def heap_path_prediction(self, initial_point, r_c, K):
        fig, ax = plt.subplots(figsize=(11, 7.166666))
        origin_x = 0
        origin_y = 0
        ax.set_xlim(origin_x-120,origin_x + 120)
        ax.set_ylim(origin_y-140, origin_y + 20)
        plt.ion()

        ############################
        # Initialize the heap
        paths = []  # Start with a probability of 1.0
        neighbours = self.find_closest_neighbours(initial_point, r_c*2)
        print(f"Number of neighbours: {len(neighbours)}")
        if not neighbours:
            raise ValueError("No neighbours found initially!!")         
            
        predicted_courses, average_distance, probabilities_list = self.compute_probabilistic_course(neighbours)

        most_probable_index = np.argsort(probabilities_list)[-3:][::-1]        
        for index in most_probable_index:
            pred_course = predicted_courses[index]
            new_point = predict_next_point(pred_course, average_distance, initial_point)
            if not check_point_within_bounds(new_point):
                return
            new_path = [initial_point, new_point]
            new_prob = probabilities_list[index]
            heapq.heappush(paths, (-new_prob, new_path))
        probabilties = [path[0] for path in paths]
        print(f"{len(paths)} number of initial paths, with probabilities: {probabilties}")
        # temp_in = input("Press enter to continue")
        ############################
        possible_paths = []
        THRESHOLD_PROBABILITY = 0.2
        for k in range(100):
            ax.clear()
            ax.set_xlim(origin_x-120,origin_x + 120)
            ax.set_ylim(origin_y-140, origin_y + 20)
            # Pop the path with highest probabilit
            print(f"Iteration {k}")
            updated_paths = []
            for j in range(3):
                if not paths:
                    break
                prob, path = heapq.heappop(paths)
                current_point = path[-1]
                neighbours = self.find_closest_neighbours(current_point, r_c)
                if not neighbours:
                    print("No neighbours found!!")
                    break
                predicted_courses, average_distance, probabilities_list = self.compute_probabilistic_course(neighbours)
                most_probable_index = np.argsort(probabilities_list)[-3:][::-1]
                for index in most_probable_index:
                    # if probabilities_list[index] < THRESHOLD_PROBABILITY:
                    #     break
                    pred_course = predicted_courses[index]
                    new_point = predict_next_point(pred_course, average_distance, current_point)
                    if not check_point_within_bounds(new_point,plot=False):
                        print("Point out of bounds!!")
                        new_path = path + [new_point]
                        new_prob = probabilities_list[index] * prob
                        heapq.heappush(possible_paths, (new_prob, new_path))
                        # temp_in = input("Press enter to continue")
                        break
                    new_path = path + [new_point]  #[path, new_point]
                    new_prob = probabilities_list[index] * prob
                    # if len(updated_paths) == 0:
                    #     heapq.heappush(updated_paths, (new_prob, new_path))

                    similar = False
                    for existing_path in updated_paths:
                        if check_similarites_between_tracks(new_path, existing_path):
                            similar = True
                            break

                    if not similar:
                        heapq.heappush(updated_paths, (new_prob, new_path))

           
            if updated_paths:
                # unique_paths = []
                # for i in range(len(updated_paths)):
                #     if any(check_similarites_between_tracks_2(updated_paths[i][1], updated_paths[j][1]) for j in range(i+1, len(updated_paths))):
                #         continue  # Skip this path if it's similar to any other path
                #     unique_paths.append(updated_paths[i])  # If it's not similar to any other path, add it to the unique paths

                # paths = unique_paths
                # paths = []
                # paths_to_remove_index = []
                # for i in range(len(updated_paths)):
                #     for j in range(i+1, len(updated_paths)):
                #         if j in paths_to_remove_index:
                #             continue
                #         if check_similarites_between_tracks_2(updated_paths[i][1], updated_paths[j][1]):
                #             temp_in = input("Press enter to continue")
                #             # lengths = [len(updated_paths[i][1]), len(updated_paths[j][1])]
                #             # if lengths[0] > lengths[1]:
                #             paths_to_remove_index.append(j)
                #         else:
                #             paths.append(updated_paths[i])
            

                paths = updated_paths
            else:
                break

            
           
            for path in paths:
                # Plot each path
                points = path[1]
                # print(f"Path: {points}")
                xs = []
                ys = []
                for point in points:
                    # print(f"Point: {point}")
                    xs.append(point[0])
                    ys.append(point[1])
                ax.plot(xs, ys, '-o', label='Path')

            ax.legend()  # Add legend if needed
            plt.draw()
            plt.pause(0.1)  # Pause to update the plot
            if k ==15:
                print(paths)
                print(len(paths))
                temp_in = input("Press enter to continue")
            # if k==7:
            #     print(paths)
            #     temp_in = input("Press enter to continue")
            # if len(paths) < 2:
            #     print(f"PATHS: {paths}")
            #     print(f"Number of paths: {len(paths)}")
            #     temp_in = input("Press enter to continue")


        # return_paths = []
        # for _ in range(3):
        #     if not paths:
        #         break
        #     prob, path = heapq.heappop(paths)
        #     return_path = []
        #     for point in path:
        #         return_path.append(point[:2])
        #     return_paths.append(path)

        # return return_paths
        return_paths = []
        while possible_paths:
            prob, path = heapq.heappop(possible_paths)
            return_path = []
            for point in path:
                return_path.append(point[:2])
            return_paths.append(return_path)
        return return_paths


    def run_prediction(self, initial_point):
        # pred_paths = self.iterative_path_prediction(initial_point, self.r_c, self.K)
        pred_paths = self.heap_path_prediction(initial_point, self.r_c, self.K)
        # print(pred_paths)
        # temp_in = input("Press enter to continue")
        for pred_path in pred_paths:
            ax, origin_x, origin_y = start_plot()
            ax, origin_x, origin_y, legend_elements = plot_all_vessel_tracks(ax, self.X_B, origin_x, origin_y,
                                                                            save_plot=False)
            ax, origin_x, origin_y, legend_elements = plot_predicted_path(ax, pred_path, initial_point, self.r_c, self.K, origin_x, origin_y,
                                legend_elements, save_plot=False)
            if self.compare_to_track:
                plot_single_vessel_track(ax, self.track, origin_x, origin_y, legend_elements, self.track_id, save_plot=True)
            plt.close()

if __name__ == '__main__':
    path_predictor = NCDM('npy_files/X_B_filtered.npy')
    print(path_predictor.num_tracks)

    # for i in range(path_predictor.num_tracks):
    #     initial_point = path_predictor.find_track(i)
    #     path_predictor.run_prediction(initial_point)

    initial_point = path_predictor.find_track(0)        
    path_predictor.run_prediction(initial_point)

    # # initial_point = [-23, -105]
    # # random_angle = -10
    # # initial_point = find_initial_points(initial_point, random_angle)
    
