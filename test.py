import matplotlib.pyplot as plt
import numpy as np

def calculate_course(p1, p2):
    """Calculate the course from point p1 to p2."""
    # dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dx, dy))

def predict_next_point(average_course, average_distance, current_point):
    """Predict the next point based on average course and distance."""
    x2, y2 = current_point[2], current_point[3]
    x3 = x2 + average_distance * np.sin(np.radians(average_course))
    y3 = y2 + average_distance * np.cos(np.radians(average_course))
    return [x2, y2, x3, y3]

X_B = np.load('npy_files/X_B.npy', allow_pickle=True).item()

fig, ax = plt.subplots(figsize=(11, 7.166666))

for track_id, track in X_B.items():
    for sub_track in track:
        print(sub_track)
        for i in range(0,len(sub_track),2):
            x_point = sub_track[i]
            y_point = sub_track[i+1]
            ax.plot(x_point,y_point,marker='o',color='red')
        course = calculate_course(sub_track[0:2],sub_track[2:4])
        print(course)
        next_point = predict_next_point(course, 2, sub_track)
        ax.plot(next_point[2],next_point[3],marker='o',color='blue')
        plt.show()
        temp_in = input("Press Enter to continue...")

