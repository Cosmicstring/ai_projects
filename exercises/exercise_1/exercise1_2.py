import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from lib.clustering import MyMeanShift
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestCentroid

from sklearn.cluster import MeanShift, estimate_bandwidth

from lib.parser import ArgParserEx12
from lib.error_handling import levels

argparse = ArgParserEx12()
args = argparse.parse_args()

def gaussian_kernel_diagcov(h, x):
    d = np.ndim(x) + 1
    return 1./(np.pi**d * h**d) * np.exp(-0.5 * np.linalg.norm(x/h, axis=-1)**2)

if __name__ == "__main__":

    csvdata = np.loadtxt(args.csvpath, delimiter=args.csvdelimiter)

    # Grab only the X,Y coordinates

    X,Y = csvdata[:,1], csvdata[:,2]

    # Let's just rescale all the data points to (0,1) range, to avoid any numerical issues
    scaler = MinMaxScaler()
    X, Y = scaler.fit_transform(np.array((X, Y)).T).T

    data = np.stack((X, Y), axis=-1)
    ranges = np.asarray(2*(scaler.feature_range,), dtype=np.float32)

    meanshiftalgo = MyMeanShift(data,
                                ranges,
                                visualize=args.visualize,
                                bandwidth=args.bandwidth)

    # Set the callable kernel to use -> in this case Gaussian, with diagonal covariance
    params = [args.bandwidth]

    meanshiftalgo.set_kernel(gaussian_kernel_diagcov, params)

    # Find the centroids
    centroids = meanshiftalgo.find_maxima()

    # Now find closest centroids
    classes = np.arange(centroids.shape[0])

    # Default is euclidian norm as requested
    fcntrd = NearestCentroid()
    nctrd = fcntrd.fit(centroids, classes)

    dataclasses = nctrd.predict(data)

    fig, ax = plt.subplots(2,1, figsize=(10,12))
    for i, centroid in enumerate(centroids):
        ax[0].plot(centroid[0], centroid[1],
                   color = cm.tab20(i),
                   marker='*',
                   markersize=15)
    for dclass, datapoint in zip(dataclasses, data):
        ax[0].plot(datapoint[0], datapoint[1],
                   color = cm.tab20(dclass),
                   marker='o',
                   alpha=.5)

    ax[0].set_xlabel("Axis 1")
    ax[0].set_ylabel("Axis 2")
    ax[0].set_title("My MeanShift algorithm")

    # Now plot the result of the sklearn MeanShift

    bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    for k in zip(range(n_clusters_)):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        ax[1].plot(data[my_members, 0], data[my_members, 1], 'o',
                   color=cm.tab20(k), alpha=.5)
        ax[1].plot(
            cluster_center[0],
            cluster_center[1],
            "*",
            color=cm.tab20(k),
            markersize=15)

    ax[1].set_title("sklearn's MeanShift algorithm")
    ax[1].set_xlabel("Axis 1")
    ax[1].set_ylabel("Axis 2")

    fig.tight_layout()
    fig.savefig("clusters_exercise_1_2.pdf")
