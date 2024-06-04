from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime
import os

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
    GMM = GaussianMixture(n_components=1,random_state=0).fit(X)
    
    while k <= max_comps:
        prev_GMM = GMM
        try:
            GMM = GaussianMixture(n_components=k,random_state=0).fit(X)
        except Exception as e:
            print(f"Failed to fit GMM with {k} components: {e}")
            print(f"Size of data: {len(X)}")
            GMM = prev_GMM
            break
       
        for i in range(k-1):
            for j in range(i+1, k):
                if np.linalg.norm(GMM.means_[i] - GMM.means_[j]) < margin:
                    # Demonstation of GMM components selection
                    # gmm_plot(X, GMM, prev_GMM, save_plot=False)
                    GMM = prev_GMM
                    found = True
                    break
            if found:
                break
        if found:
            break

        k += 1
    
    # print(f"Number of components: {GMM.n_components}")
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

def gmm_plot(data, gmm, prev_gmm, save_plot=False):
    #  colors = ['#ff7f0e','#1f77b4', '#2ca02c','#c73838','#c738c0',"#33A8FF",'#33FFBD']  # Orange, blå, grønn, rød, rosa, lyse blå, turkis
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20, 7.166666))
    ax1.hist(data, bins=100, density=True, color='#1f77b4')
    ax2.hist(data, bins=100, density=True, color='#1f77b4')

    red_color = "#c73838"
    font_size = 17

    x = np.linspace(-360, 360, 1000)
    y_poses = []
    for i in range(gmm.n_components):
        max_pdf_point = gmm.means_[i, 0]
        max_pdf_value = norm.pdf(max_pdf_point, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0]))
        prob = gmm.weights_[i]
        x_pos = max_pdf_point
        y_pos = max_pdf_value
        y_poses.append(y_pos)

        diff = 0
        if x_pos < 0:
            diff = abs(x_pos*0.05)
        elif x_pos > 0:
            diff = -abs(x_pos*0.05)

        ax2.plot(x, norm.pdf(x, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0])), lw="3", color='#c73838')
        ax2.plot(max_pdf_point, max_pdf_value, color=red_color, marker='o', markersize=10)
        ax2.text(x_pos + diff , y_pos*1.03, f"\u03BC = {max_pdf_point:.1f}", fontsize=font_size, verticalalignment='bottom', horizontalalignment='center')


    gmm = prev_gmm
    y_poses_prev = []
    for i in range(gmm.n_components):
        max_pdf_point = gmm.means_[i, 0]
        max_pdf_value = norm.pdf(max_pdf_point, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0]))
        prob = gmm.weights_[i]
        x_pos = max_pdf_point
        y_pos = max_pdf_value
        y_poses_prev.append(y_pos)

        diff = 0
        if x_pos < 0:
            diff = abs(x_pos*0.05)
        elif x_pos > 0:
            diff = -abs(x_pos*0.05)

        ax1.plot(x, norm.pdf(x, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0])), lw="3", color='#c73838')
        ax1.plot(max_pdf_point, max_pdf_value, color=red_color, marker='o', markersize=10)
        ax1.text(x_pos + diff , y_pos*1.03, f"\u03BC = {max_pdf_point:.1f}", fontsize=font_size, verticalalignment='bottom', horizontalalignment='center')   

    if max(y_poses) > max(y_poses_prev):
        max_y = max(y_poses)
    else:
        max_y = max(y_poses_prev)
    ax1.set_ylim([0, max_y*1.1])
    ax2.set_ylim([0, max_y*1.1])

    ax1.tick_params(axis='both', which='major', labelsize=font_size*1.33)
    ax2.tick_params(axis='both', which='major', labelsize=font_size*1.33)
       
    ax1.set_xlabel('Courses of close neighbors  [\u00B0]', fontsize=font_size*1.33)
    ax1.set_ylabel('Density', fontsize=font_size*1.33)
    ax2.set_xlabel('Courses of close neighbors  [\u00B0]', fontsize=font_size*1.33)
    ax2.set_ylabel('Density', fontsize=font_size*1.33)
    plt.tight_layout()

    if save_plot:
        save_path = os.path.dirname(os.path.realpath(__file__))
        now_time = datetime.datetime.now().strftime("%H,%M,%S")
        save_path = os.path.join(save_path, f'Images/Model_selection/plot_({now_time}).png')
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)