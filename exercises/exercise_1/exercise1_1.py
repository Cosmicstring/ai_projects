import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler

import logging

from lib.parser import ArgParserEx11
from lib.error_handling import levels

argparse = ArgParserEx11()
args = argparse.parse_args()

def get_heatmap(x, y, s, bins=(512,512)):

    # Make a 2Dhistogram for heatmap
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)

    # Smooth the histogram by the gaussian filter
    heatmap = gaussian_filter(heatmap, sigma=s)

    return heatmap.T


if __name__ == "__main__":

    csvdata = np.loadtxt(args.csvpath, delimiter=args.csvdelimiter)
    heinekenimg = cv2.imread(args.imagepath)

    # NOTE:
    #
    # Need to make sure points are properly rescaled to image width - height, not only for plotting
    # reasons, but also to properly transform the 2D point pdf, applying correct Jacobian
    xscale = (0, heinekenimg.shape[1])

    # Trick for matching the image layout along the y-axis, i.e. (0,0) being in the upper left corner
    yscale = (-heinekenimg.shape[0], 0)

    Xscaler = MinMaxScaler(xscale)
    Yscaler = MinMaxScaler(yscale)

    scaledXdata = Xscaler.fit_transform(csvdata[:,1].reshape(-1,1))
    scaledYdata = Yscaler.fit_transform(csvdata[:,2].reshape(-1,1))

    # Reverse back the reflection transform, for plotting
    rescaledPoints = np.stack((scaledXdata.flatten(), (-scaledYdata).flatten()), axis=-1)

    fig, ax = plt.subplots(1,1)

    # On the same fig show the heineken image
    ax.imshow(heinekenimg)

    if args.level == levels.DEBUG:
        # If we want to go to DEBUG mode, then also scatter the points
        ax.scatter(rescaledPoints[:,0],rescaledPoints[:,1], marker='o', s=3, alpha=.5)

    # Find the heatmap using gaussian kernel with size set by args.gaussian_kernel_size
    heatmap = get_heatmap(rescaledPoints[:,0],
                          rescaledPoints[:,1],
                          args.gaussian_kernel_size,
                          bins=heinekenimg.shape[:-1][::-1])

    # Don't plot the 0-values pixels
    masked_heatmap = np.ma.masked_where(heatmap < args.heatmap_threshold, heatmap)
    cb = ax.imshow(masked_heatmap, cmap="jet", alpha=.8)

    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(cb, ax=ax, pad=0.05)
    fig.tight_layout()

    # Save to vectorized format, without any extra white background than needed
    plt.savefig("heatmap_exercise_1_1.pdf", bbox_inches="tight", dpi=300)
