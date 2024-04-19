
import matplotlib.pyplot as plt
import numpy as np



def plot_AIS(ax,X_B):
    points = []
    for k,key in enumerate(X_B):
        points.append(X_B[key][:,0:2])
        # if k == 2:
            # break
    points = np.array(points, dtype=object)
    points = np.concatenate(points, axis=0)  # concatenate all arrays into one
    # print(points)
    # plot the points
    # plt.figure(figsize=(10,10))
    ax.set_xlim(-120,120)
    ax.set_ylim(-140,20)
    ax.scatter(points[:,0],points[:,1],s=0.01,marker=".")
    ax.grid()
    plt.savefig('Images/plot.png',dpi=300)

def plot_single_track(ax,track_id,X_B):
    track = X_B[track_id]
    points = track[:,0:2]
    ax.plot(points[:,0],points[:,1],color='red')
    ax.plot(points[0,0],points[0,1],marker='o',color='green')
    ax.set_xlim(-120,120)
    ax.set_ylim(-140,20)
    plt.savefig('Images/single_track.png',dpi=300)

def plot_predicted_path(ax,point_list):
    point_list = np.array(point_list)
    ax.plot(point_list[:,0],point_list[:,1],color='red')
    plt.savefig('Images/predicted_path.png',dpi=300)


def course(p1,p2):
    course = (np.arctan2(p2[1]-p1[1],p2[0]-p1[0]))
    course = np.degrees(course)
    return course

def distance(p1,p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def closest_neighbour(X_B,point,r_c):
    # first_init_point = [point[0], point[1]]
    second_point = [point[2], point[3]]
    # 
    closest_neighbours = {}
    for track_id, track in X_B.items():
        # print(track_id)
        # print(track)
        for sub_track in track:
            neighbours = True
            for i in range(0,len(sub_track),2):
                x_point = sub_track[i]
                y_point = sub_track[i+1]
                # print("X_point")
                # print(x_point)
                # print("Y_point")
                # print(y_point)
                # print("Distance")
                # print(second_init_point)
                # print(distance(second_init_point,[x_point,y_point]))
                if distance(second_point,[x_point,y_point]) > r_c:
                #     # print('In')
                #     print([x_point,y_point])
                # else:
                    neighbours = False
                    break
            if neighbours:
                if track_id in closest_neighbours:
                    closest_neighbours[track_id].append(sub_track)
                else:    
                    closest_neighbours[track_id] = [sub_track]
    # print(closest_neighbours)
    return closest_neighbours

def closest_neighbour_within_course_func(closest_neighbours,point,delta_course):
    x1 = point[0]
    x2 = point[2]
    y1 = point[1]
    y2 = point[3]
    course_of_current_point = course([x1,y1],[x2,y2])
    # print(f"Course of current point = {course_of_current_point}")
    closest_neighbours_within_course = {}
    closest_neighbours_within_course_info = {}
    for k,(neighbour_id, neighbours) in enumerate(closest_neighbours.items()):
        for neighbour in neighbours:
            course_of_neighbour = course([neighbour[2],neighbour[3]],[neighbour[4],neighbour[5]])
            if abs(course_of_neighbour - course_of_current_point) < delta_course:
                # print(f"Neighbour ID = {neighbour_id}, Course = {course_of_neighbour}")
                distance_between_points = distance([neighbour[2],neighbour[3]],[neighbour[4],neighbour[5]])
                if neighbour_id in closest_neighbours_within_course:
                    closest_neighbours_within_course[neighbour_id].append(neighbour)
                    closest_neighbours_within_course_info[neighbour_id].append([course_of_neighbour,distance_between_points])
                else:
                    closest_neighbours_within_course[neighbour_id] = [neighbour]
                    closest_neighbours_within_course_info[neighbour_id] = [[course_of_neighbour,distance_between_points]]
        # if k > 60:
        #     break
    # for k,(neighbour_id, neighbours) in enumerate(closest_neighbours_within_course.items()):
    #     print(neighbour_id)
    #     print(neighbours)
    # for k,(neighbour_id, neighbours) in enumerate(closest_neighbours_within_course_info.items()):
    #     print(neighbour_id)
    #     print(neighbours)
    return closest_neighbours_within_course, closest_neighbours_within_course_info
    #     neighbour_course = course([neighbour[0],neighbour[1]],[neighbour[2],neighbour[3]])
    #     if abs(neighbour_course - init_course) < delta_course:
    #         closest_neighbours_within_course.append(neighbour)
    # return closest_neighbours_within_course

def average_course_and_distance_func(closest_neighbour_within_course_info):
    average_course = 0
    average_distance = 0
    counter = 0
    for k,(neighbour_id, infos) in enumerate(closest_neighbour_within_course_info.items()):
        for info in infos:
            average_course += info[0]
            average_distance += info[1]
            counter += 1
    average_course = average_course / counter
    average_distance = average_distance / counter
    # print(f"Average Course = {average_course}")
    # print(f"Average Distance = {average_distance}")
    return average_course, average_distance
        
def predict_next_point_func(average_course,average_distance,point):
    x1 = point[0]
    x2 = point[2]
    y1 = point[1]
    y2 = point[3]
    # course_of_current_point = course([x1,y1],[x2,y2])
    # print(f"Course of current point = {course_of_current_point}")
    x3 = x2 + average_distance*np.cos(np.radians(average_course))
    # print(x2,x3)
    y3 = y2 + average_distance*np.sin(np.radians(average_course))
    # print(y2,y3)
    # print(f"Next Point = {x3},{y3}")
    return [x2,y2,x3,y3]

def test(X_B):
    init_point = X_B[0][0][0:4]
    init_point = [65.7, -35.3, 61.8, -36.5]
    single_init_point = [-20, -20]
    direction = -130
    init_point = [single_init_point[0],single_init_point[1],single_init_point[0]+np.sin(np.radians(direction)),single_init_point[1]+np.cos(np.radians(direction))]
    # init_point = [-20, -20, -21, -21]
    # [x1, y1, x2, y2] 
    # Increase x2 more to east, decrease x2 to go west
    # Increase y2 to go north, decrease y2 to go south
    print('Init Point')
    print(init_point)
    K = 100
    Jmax = 200
    J_n = np.ones(K)*Jmax
    J_n[0] = 1
    N_kj = np.ones(K)
    N_kj[0] = Jmax
    r_c = 10
    alpha = 100
    sigma = 0.1
    delta_course = 15 # degrees

    # print(course([init_point[0],init_point[2]],[init_point[1],init_point[3]]))
    point_list = []
    point_list.append(init_point[0:2])
    next_point = init_point
    k=0
    while k<K:
        print(f"K={k}")
        print(f"Next Point = {next_point[-2:]}")
        closest_neighbours = closest_neighbour(X_B,next_point,r_c)
        # for k,neighbour in enumerate(closest_neighbours):
        #     print(neighbour)
        closest_neighbour_within_course, closest_neighbour_within_course_info = closest_neighbour_within_course_func(closest_neighbours,next_point,delta_course)
        if len(closest_neighbour_within_course) == 0:
            break
            # if r_c > 15:
            #     break
            # r_c += 1
            # continue
        average_course, average_distance = average_course_and_distance_func(closest_neighbour_within_course_info)
        # print(f"No of Neighbours = {len(closest_neighbours)}, Average Course = {average_course}, Average Distance = {average_distance}")
        next_point = predict_next_point_func(average_course,average_distance,next_point)
        point_list.append(next_point[0:2])
        k += 1
    point_list.append(next_point[2:4])
    print(f"Ended at K={k} iterations")
    # print(point_list)
    return point_list
    # point_list = np.array(point_list)
    # fig, ax = plt.subplots()
    # plt.plot(point_list[:,0],point_list[:,1],marker='o',color='red')
    # plt.savefig('Images/predicted_path.png',dpi=300)
    # print(closest_neighbour_within_course.keys())
    # print(len(closest_neighbour_within_course.keys()))
    # fig2, ax2 = plt.subplots()
    # plot_single_track(ax2,37,X_B)
    # now i want to check the course of the closest neighbour




    # for k in range(K):
    #     q = 0
    #     print(f"k={k}")
    #     for j in range(int(J_n[k])):
    #         # Find closes neighbour

    #         print(j)
    #     print('Done')
    #         # print(X_B[k][j][0:4])
    #         # print(init_point)
    #         # print(np.linalg.norm(X_B[k][j][0:4]-init_point))
    #         # if np.linalg.norm(X_B[k][j][0:4]-init_point) < r_c:
    #         #     N_kj[k] += 1
    #         #     print('In')

if __name__ == '__main__':
    X_B = np.load('X_B.npy',allow_pickle=True).item()
    fig, ax = plt.subplots()
    plot_AIS(ax, X_B)
    init_point = X_B[0][0][0:4]
    print('Init Point')
    print(init_point)
    # x1 = init_point[0]
    # x2 = init_point[2]
    # y1 = init_point[1]
    # y2 = init_point[3]
    # ax.plot([init_point[0],init_point[2]],[init_point[1],init_point[3]],marker='o',color='red')
    # plt.savefig('Images/plot.png',dpi=300)
    pred_path = test(X_B)
    plot_predicted_path(ax,pred_path)
