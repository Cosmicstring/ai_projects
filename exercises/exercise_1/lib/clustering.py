import numpy as np
from functools import partial

from scipy.spatial import cKDTree

import matplotlib.pyplot as plt

class MyMeanShift(object):

    def __init__(self,
                 data,
                 ranges,
                 bandwidth = 8,
                 climbers = 16,
                 max_iter = 1000,
                 mn_threshold = 1e-3,
                 visualize = False
                 ):
        """
        MeanShift algorithm for finding cluster centers. Implemented according to [1].

        ...

        Parameters
        ----------
            data           : ndarray
                Numpy array containing all the data points, shape (N,d), where
                N is the number of data points and d the dimensionality of the space

            ranges         : ndarray
                The range of values for the coordinates along each axis, with shape (d,2),
                with ranges[i,0] representing the minimum value for coordinate along axis i,
                and ranges[i,1] representing the maximum value for axis i

            bandwidth       : float
                The bandwidth to be used for assigning weights with the kernel approximation

            climbers       : int, optional
                Number of probing points to be used for detecting local maxima of the
                data pdf

            visualize      : bool, optional
                Whether to plot and follow how the mean shift algorithm progresses

            max_iter       : int, optional
                Maximum number of iterations to try before stopping the algorithm

            mn_threshold   : float, optional
                Threshold value for the magnitude of the mn vector, if below this, don't move the given point

        [1] Comaniciu, Dorin, and Peter Meer. "Mean shift: A robust approach toward feature space analysis." IEEE Transactions on pattern analysis and machine intelligence 24.5 (2002): 603-619.
        """

        # Check that the ranges has correct shape
        if ranges.shape[0] != np.ndim(data):
            raise ValueError("The number of dimensions provided in ranges doesn't match with dimensionality of data!")

        self.data     = data
        self.bandwidth = bandwidth
        self.climbers = climbers
        self.visualize = visualize
        self.max_iter = max_iter
        self.mn_threshold = mn_threshold

        # Make the KDTree of the dataset, to be used later, distance
        # by default Euclidian!

        # FIXME: See whether there are some KDTree structures in python allowing for
        #        arbitrary non-flat metric to be used as a distance measure
        self.dataKDTree = cKDTree(self.data)

        self.kfunc      = None

        # Initialize the starting points, by uniformly sampling the ranges along each axis
        self.initial_points = np.random.uniform(
            low   = ranges[:,0],
            high  = ranges[:,1],
            size  = (self.climbers, np.ndim(self.data)),).astype(np.float32)

    def find_maxima(self):

        if (self.kfunc is None):
            raise ValueError("You need to set kfunc using 'self.set_kernel' before starting this method!")

        def _get_mn(positions):

            mn = np.zeros(positions.shape, dtype=positions.dtype)

            for i, pos in enumerate(positions):
                _dx = (pos - self.data)
                _tmp = np.sum((self.data.T * self.kfunc(_dx)).T, axis=0) / np.sum(self.kfunc(_dx))
                mn[i] = self.bandwidth*(_tmp - pos)

            return mn

        def _apply_one_step(positions):

            # Get the mean shift vectors
            mn = _get_mn(positions)

            # If the whole mn norm is below the threshold, stop execution
            _mn_norms = np.linalg.norm(mn,axis=1)
            belowthrshld = _mn_norms < self.mn_threshold

            return positions + mn, np.alltrue(belowthrshld)

        _iter = 0
        _positions = self.initial_points
        _cond = False

        if self.visualize:
            fig, ax = plt.subplots()

        while (_iter < self.max_iter and not _cond):

            if _iter%10==0:
                if self.visualize:
                    # For visualization purposes
                    ax.scatter(self.data[:,0], self.data[:,1], marker='o', c='b', alpha=.1)
                    for centroid in _positions:
                        ax.scatter(centroid[0], centroid[1], marker='x', c='k', s=15, alpha=.1)
                    ax.set_xlabel("Axis 1")
                    ax.set_ylabel("Axis 2")
                    fig.tight_layout()
                    plt.savefig(f"figs/step_{_iter}.png")

            _positions, _cond = _apply_one_step(_positions)
            _iter +=1

        # Once finished prune the final positions to find selected maxima

        return self.prune_pos(_positions)


    def prune_pos(self, fpos):
        """
        Method to prune the remaining points after the mean shift iterations are finished.

        Parameters:
        -----------
        fpos      : ndarray
            Set of final positions of the initial set of climbers

        Returns:
        --------
        centroids : ndarray
            The list of unique centroids found
        """

        # Now assign weights to the fpos, based on how many points are within the
        # radius=bandwidth ball
        weights = np.zeros(fpos.shape[0], dtype=np.int)

        for i, pos in enumerate(fpos):
            weights[i] = self.dataKDTree.query_ball_point(pos,
                                                          self.bandwidth,
                                                          return_length=True)

        # Set the boolean array for centroid classes
        unique = np.ones(fpos.shape[0], dtype=np.bool_)

        # Now make a KDTree for the centroids to find unique ones
        centroidKDTree = cKDTree(fpos)

        # Now sort according to their weight --> descending
        ind = np.argsort(weights)[::-1]

        for i in ind:
            pos = fpos[i]
            if unique[i]:
                # Find all fpos within the ball of radius=bandwidth
                within = centroidKDTree.query_ball_point(pos,
                                                         self.bandwidth)

                # Set all to not unique except the one with the maximum weight
                unique[within] = False
                unique[i] = True

        # Now only return the unique positions
        return fpos[unique]

    def set_kernel(self,
                   kfunc,
#                   kfuncgrad,
                   kparams):
        """
        Kernel to be used in smoothing the point for estimating weights needed for performing
        mean shift steps.

        Parameters:
        -----------

        kfunc       : callable
            A callable function parameterized as kfunc(params, x), where params corresponds to the 'kparams'
            passed here, and 'x' is the same dimensionality as the data space
        kparams     : list
            A keyword dictionary to be passed as parameter argument to 'kfunc'

        Returns:
        --------
        None
        """

        self.kfunc = partial(kfunc, *kparams)
