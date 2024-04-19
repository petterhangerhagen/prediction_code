
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

def course(p1,p2):
    course = (np.arctan2(p2[1]-p1[1],p2[0]-p1[0]))
    course = np.degrees(course)
    return course

def distance(p1,p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def closest_neighbour(X_B,init_point,r_c):
    first_init_point = [init_point[0], init_point[1]]
    second_init_point = [init_point[2], init_point[3]]
    # 
    closest_neighbours = {}
    for track_id, track in X_B.items():
        print(track_id)
        print(track)
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
                if distance(second_init_point,[x_point,y_point]) > r_c:
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

def closest_neighbour_within_course(closest_neighbours,init_point,delta_course):
    init_course = course([init_point[0],init_point[2]],[init_point[1],init_point[3]])
    closest_neighbours_within_course = []
    for neighbour in closest_neighbours:
        neighbour_course = course([neighbour[0],neighbour[1]],[neighbour[2],neighbour[3]])
        if abs(neighbour_course - init_course) < delta_course:
            closest_neighbours_within_course.append(neighbour)
    return closest_neighbours_within_course
        
def test(X_B):
    init_point = X_B[0][0][0:4]
    print('Init Point')
    print(init_point)
    K = 3
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
    
    
    closest_neighbours = closest_neighbour(X_B,init_point,r_c)
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
    ax.plot([init_point[0],init_point[2]],[init_point[1],init_point[3]],marker='o',color='red')
    plt.savefig('Images/plot.png',dpi=300)
    test(X_B)
