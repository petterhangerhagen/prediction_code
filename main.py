
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
    plt.savefig('plot.png',dpi=300)

def test(X_B):
    init_point = X_B[0][0][0:4]
    print(init_point)

if __name__ == '__main__':
    X_B = np.load('X_B.npy',allow_pickle=True).item()
    fig, ax = plt.subplots()
    plot_AIS(ax, X_B)
    test(X_B)
