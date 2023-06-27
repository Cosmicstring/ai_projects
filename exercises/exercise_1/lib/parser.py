import argparse
import configparser

class ArgParserEx1(argparse.ArgumentParser):

    def __init__(self, description="Argparser for the exercise 1"):
        super().__init__(description=description)

        self.add_argument(
            "--imagepath",
            type=str, default="Heineken.jpg"
        )

        self.add_argument(
            "--csvpath",
            type=str, default="Heineken.csv"
        )

        self.add_argument(
            "--csvdelimiter",
            type=str, default=";"
        )

        self.add_argument(
            "--level",
            type=int, default=20,
            choices=[0,10,20,30,40,50]
        )

class ArgParserEx11(ArgParserEx1):

    def __init__(self):

        super().__init__(description="Argparser for the exercise 1.1")

        self.add_argument(
            "--gaussian_kernel_size",
            type = int, default=12,
            help = "The size of Gaussian kernel to be used when smoothing the scattered points using the scipy.ndimage.filters.gaussian_filter"
        )

        self.add_argument(
            "--heatmap_threshold",
            type = float, default=2e-3,
            help = "The heatmap threshold to be used for masking out all pixels values below it"
        )

class ArgParserEx12(ArgParserEx1):

    def __init__(self):

        super().__init__(description="Argparser for the exercise 1.2")

        self.add_argument(
            "--bandwidth",
            type = float, default=0.08,
            help = "The bandwith of the mean shift algorithm"
        )

        self.add_argument(
            "--climbers",
            type = int, default=1024,
            help = "Whether to plot the progress of mean shift algorithm for visualization"
        )

        self.add_argument(
            "--visualize",
            type = bool, default=0,
            help = "Whether to plot the progress of mean shift algorithm for visualization"
        )
