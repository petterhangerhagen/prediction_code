from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

def get_GMM(X, max_comps, margin):
    # Initialize variables
    k = 2
    found = False
    # Fit initial GMM with 1 component
    GMM = GaussianMixture(n_components=1).fit(X)
    
    while k <= max_comps:
        prev_GMM = GMM
        try:
            GMM = GaussianMixture(n_components=k).fit(X)
        except Exception as e:
            print(f"Failed to fit GMM with {k} components: {e}")
            GMM = prev_GMM
            break
       
        for i in range(k-1):
            for j in range(i+1, k):
                if np.linalg.norm(GMM.means_[i] - GMM.means_[j]) < margin:
                    GMM = prev_GMM
                    found = True
                    break
            if found:
                break
        if found:
            break

        k += 1

    return GMM

# if __name__ == "__main__":
#     mode = "MU"
#     max_comps = 5
#     X = np.load("data.npy")
#     # print(X)
#     courses = X[:, 0].reshape(-1, 1)
#     # print(courses)
    
#     GMM = get_GMM(courses, max_comps, mode)
#     print(GMM.means_)
#     print(GMM.weights_)
#     print(GMM.get_params())
#     # print(GMM)

#     plt.hist(X, bins=50)
#     plt.show()