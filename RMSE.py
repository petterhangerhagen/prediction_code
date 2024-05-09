
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load original and predicted tracks
# original_track = np.load("npy_files/track.npy")
# predicted_track = np.load("npy_files/points_list.npy")

def RMSE(original_track, predicted_track, plot_statement=False):
    # Calculate the distance between the last point of the original track and each point in the predicted track
    distances = np.linalg.norm(original_track[-1] - predicted_track, axis=1)

    # Find the index of the point in the predicted track that is closest to the end point of the original track
    index_to_cut = np.argmin(distances)

    # Cut the predicted track short
    predicted_track = predicted_track[:index_to_cut+1]

    # Interpolate predicted track to fit original track
    interp_func = interp1d(np.linspace(0, 1, len(predicted_track)), predicted_track, axis=0)
    interpolated_predicted_track = interp_func(np.linspace(0, 1, len(original_track)))

    if plot_statement:
        original_track = np.array(original_track)
        predicted_track = np.array(predicted_track)
        interpolated_predicted_track = np.array(interpolated_predicted_track)
        # Plot original track
        fig6, ax6 = plt.subplots()
        ax6.plot(original_track[:, 0], original_track[:, 1], "-o", label='Original Track', color='blue')

        # Plot predicted track
        ax6.plot(predicted_track[:, 0], predicted_track[:, 1], "-o", label='Predicted Track', color='red')

        # Plot interpolated predicted track
        ax6.plot(interpolated_predicted_track[:, 0], interpolated_predicted_track[:, 1], "-o", label='Interpolated Predicted Track', color='orange')

        ax6.set_xlabel('Longitude')
        ax6.set_ylabel('Latitude')
        ax6.set_title('Original Track and Interpolated Predicted Track')
        ax6.legend()
        ax6.grid(True)
        plt.show()

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(original_track, interpolated_predicted_track))
    return rmse


# rmse = RMSE(original_track, predicted_track, plot_statement=True)