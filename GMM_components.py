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
                    # print(f"Found GMM with {k} components")
                    GMM = prev_GMM
                    found = True
                    break
            if found:
                break
        if found:
            break

        k += 1

    return GMM

def get_GMM_modified(X, max_comps, margin):
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
                    # print(f"Found GMM with {k} components")
                    GMM = prev_GMM
                    found = True
                    break
            if found:
                break
        if found:
            break

        k += 1
    
    if not found:
        GMM = GaussianMixture(n_components=3).fit(X)

    print(f"Number of components: {GMM.n_components}")
    return GMM



def choice_of_number_of_components(data):
    # Compute BIC to determine the best number of components
    bics = []
    n_components_range = range(1, 9)  # Assuming up to 8 components
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components).fit(data)
        bics.append(gmm.bic(data))
    best_n = n_components_range[np.argmin(bics)]
    return best_n

