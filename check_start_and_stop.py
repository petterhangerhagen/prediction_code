"""
Script Title: Check start and stop
Author: Petter Hangerhagen
Email: petthang@stud.ntnu.no
Date: February 27, 2024
Description: This script checks if a track starts and stops in a specific area. The areas are defined by rectangles.
It also finds the average length of the tracks that starts and stops in the same area.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

class RectangleA:
    def __init__(self, bottom_left=[40,-60], top_right=[118,18]): # earlier [-120,-40], [-30,40]
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.index = 0
        self.name = "A"

    def __repr__(self):
        return f"RectangleA"

    def start_or_stop(self,x,y):
        if self.bottom_left[0] < x < self.top_right[0] and self.bottom_left[1] < y < self.top_right[1]:
            return True
        else:
            return False

class RectangleB:
    def __init__(self, bottom_left=[-38,-32], top_right=[-4,18]):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.index = 1
        self.name = "B"

    def __repr__(self):
        return f"RectangleB"

    def start_or_stop(self,x,y):
        if self.bottom_left[0] < x < self.top_right[0] and self.bottom_left[1] < y < self.top_right[1]:
            return True
        else:
            return False

class RectangleC:
    def __init__(self, bottom_left=[-118,-138], top_right=[-41,-80]):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.index = 2
        self.name = "C"

    def __repr__(self):
        return f"RectangleC"

    def start_or_stop(self,x,y):
        if self.bottom_left[0] < x < self.top_right[0] and self.bottom_left[1] < y < self.top_right[1]:
            return True
        else:
            return False

class RectangleD:
    def __init__(self, bottom_left=[-39,-138], top_right=[20,-90]):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.index = 3
        self.name = "D"

    def __repr__(self):
        return f"RectangleD"

    def start_or_stop(self,x,y):
        if self.bottom_left[0] < x < self.top_right[0] and self.bottom_left[1] < y < self.top_right[1]:
            return True
        else:
            return False

class RectangleE:
    def __init__(self, bottom_left=[-78,-78], top_right=[-40,-30]):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.index = 4
        self.name = "E"

    def __repr__(self):
        return f"RectangleE"

    def start_or_stop(self,x,y):
        if self.bottom_left[0] < x < self.top_right[0] and self.bottom_left[1] < y < self.top_right[1]:
            return True
        else:
            return False

class RectangleF:
    def __init__(self, bottom_left=[-2,-25], top_right=[38,10]):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.index = 5
        self.name = "F"

    def __repr__(self):
            return f"RectangleF"

    def start_or_stop(self,x,y):
        if self.bottom_left[0] < x < self.top_right[0] and self.bottom_left[1] < y < self.top_right[1]:
            return True
        else:
            return False


class CountMatrix:
    def __init__(self, reset=False):
        self.wokring_directory = os.getcwd()
        self.file_name = f"{self.wokring_directory}/npy_files/count_matrix.npy"
        if not os.path.exists(self.file_name):
            print("Creating count matrix")
            self.count_matrix = np.zeros((6,6))
            np.save(self.file_name, self.count_matrix)

        self.count_matrix = np.load(self.file_name)

        if reset:
            print("Resetting count matrix")
            self.count_matrix = np.zeros((6,6))

        self.unvalidated_track = 0
        self.number_of_tracks = 0
        self.number_of_tracks_on_diagonal = 0
        self.files_with_tracks_on_diagonal = []

        # # self.average_length_matrix_filename = "/home/aflaptop/Documents/radar_tracker/code/npy_files/average_length_matrix.npy"
        # self.average_length_matrix_filename = f"{self.wokring_directory}/code/npy_files/average_length_matrix.npy"
        # if not os.path.exists(self.average_length_matrix_filename):
        #     print("Creating average length matrix")
        #     self.average_length_matrix = np.zeros((6,6))
        #     np.save(self.average_length_matrix_filename, self.average_length_matrix)
            
        # self.average_length_matrix = np.load(self.average_length_matrix_filename)
        # if reset:
        #     #print("Resetting average lenght matrix")
        #     self.average_length_matrix = np.zeros((6,6))
        
        
    def check_start_and_stop(self,track_history,filename=None):
        rectangleA = RectangleA()
        rectangleB = RectangleB()
        rectangleC = RectangleC()
        rectangleD = RectangleD()
        rectangleE = RectangleE()
        rectangleF = RectangleF()
        rectangles = [rectangleA,rectangleB,rectangleC,rectangleD,rectangleE,rectangleF]  
        start_rectangle = {}
        stop_rectangle = {}
        for index, trajectory in track_history.items():
            self.number_of_tracks += 1
            x_start = track_history[index][0].posterior[0][0]
            y_start = track_history[index][0].posterior[0][2]
            x_stop = track_history[index][-1].posterior[0][0]
            y_stop = track_history[index][-1].posterior[0][2]
            distance = np.sqrt((x_start-x_stop)**2 + (y_start-y_stop)**2)
            for rectangle in rectangles:
                # Start
                if rectangle.start_or_stop(x_start,y_start):
                    start_rectangle[index] = [rectangle,distance]
                    print
                # Stop
                if rectangle.start_or_stop(x_stop,y_stop):
                    stop_rectangle[index] = [rectangle,distance]
            
            if index not in start_rectangle.keys() or index not in stop_rectangle.keys():
                self.unvalidated_track += 1
        
        for start_key in start_rectangle.keys():
            if start_key in stop_rectangle.keys():
                self.count_matrix[start_rectangle[start_key][0].index][stop_rectangle[start_key][0].index] += 1
                self.average_length_matrix[start_rectangle[start_key][0].index][stop_rectangle[start_key][0].index] += start_rectangle[start_key][1]
                if start_rectangle[start_key].index == stop_rectangle[start_key].index:
                    self.number_of_tracks_on_diagonal += 1
                    self.files_with_tracks_on_diagonal.append(filename.split("/")[-1])

        np.save(self.file_name, self.count_matrix)
        np.save(self.average_length_matrix_filename, self.average_length_matrix)

    def track_average_length(self):
        for k in range(6):
            for j in range(6):
                if self.count_matrix[k][j] != 0 and self.average_length_matrix[k][j] != 0:
                    self.average_length_matrix[k][j] = int(self.average_length_matrix[k][j]/self.count_matrix[k][j])
        np.save(self.average_length_matrix_filename, self.average_length_matrix)

    def check_start_and_stop_prediction(self, first_pred_point, last_pred_point):
        rectangleA = RectangleA()
        rectangleB = RectangleB()
        rectangleC = RectangleC()
        rectangleD = RectangleD()
        rectangleE = RectangleE()
        rectangleF = RectangleF()
        rectangles = [rectangleA,rectangleB,rectangleC,rectangleD,rectangleE,rectangleF]
        x_first = first_pred_point[0]
        y_first = first_pred_point[1]
        x_last = last_pred_point[0]
        y_last = last_pred_point[1]
        stop_rectangle = None
        start_rectangle = None
        for rectangle in rectangles:
            if rectangle.start_or_stop(x_last,y_last):
                stop_rectangle = rectangle
                # print(f"Stop in {stop_rectangle}")

            if rectangle.start_or_stop(x_first,y_first):
                start_rectangle = rectangle
                # print(f"Start in {start_rectangle}")
                
        if stop_rectangle is None:
            # print("No stop rectangle")
            self.unvalidated_track += 1

        # if stop_rectangle is not None and start_rectangle is not None:
        #     self.count_matrix[start_rectangle.index][stop_rectangle.index] += 1
        #     np.save(self.file_name, self.count_matrix)

        if start_rectangle is not None and stop_rectangle is not None:
            self.count_matrix[start_rectangle.index][stop_rectangle.index] += 1
            if start_rectangle.index == stop_rectangle.index:
                self.number_of_tracks_on_diagonal += 1
            np.save(self.file_name, self.count_matrix)
        else:
            self.unvalidated_track += 1

        # print(self.count_matrix)
        with open("info.txt", "w") as file:
            file.write(f"Number of tracks: {self.number_of_tracks}\n")
            file.write(f"Number on the diagonal: {self.number_of_tracks_on_diagonal}\n")
            file.write(f"Unvalid tracks: {self.unvalidated_track}\n")
        # print(f"Number on the diagonal: {self.number_of_tracks_on_diagonal}")
        # print(f"Unvalid tracks: {self.unvalidated_track}")


def plot_rectangles():
    rectangleA = RectangleA()
    rectangleB = RectangleB()
    rectangleC = RectangleC()
    rectangleD = RectangleD()
    rectangleE = RectangleE()
    rectangleF = RectangleF()
    rectangles = [rectangleA,rectangleB,rectangleC,rectangleD,rectangleE,rectangleF]

    fig, ax = plt.subplots(figsize=(11, 7.166666))

    names = ["A","B","C","D","E","F"]

    for rectangle in rectangles:
        ax.plot([rectangle.bottom_left[0], rectangle.top_right[0]], [rectangle.bottom_left[1], rectangle.bottom_left[1]], 'k-')
        ax.plot([rectangle.bottom_left[0], rectangle.top_right[0]], [rectangle.top_right[1], rectangle.top_right[1]], 'k-')
        ax.plot([rectangle.bottom_left[0], rectangle.bottom_left[0]], [rectangle.bottom_left[1], rectangle.top_right[1]], 'k-')
        ax.plot([rectangle.top_right[0], rectangle.top_right[0]], [rectangle.bottom_left[1], rectangle.top_right[1]], 'k-')
        ax.annotate(names[rectangle.index], (rectangle.bottom_left[0], rectangle.bottom_left[1]), fontsize=12, color='black')


    plt.show()
