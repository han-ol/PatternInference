import numpy as np


def make_bins(R_max, B):
    "Return bin edges for B equidistant bins between 0 and R_max."
    return np.linspace(0, R_max, B + 1)


def histogram_from_distances(distances, R_max, B):
    "Crop at R_max and bin distances (regardless of subsampling or shape) into fixed bins."
    rds = distances.reshape(-1)
    rds = rds[rds < R_max]
    hist, bin_edges = np.histogram(rds, bins=make_bins(R_max, B), density=True)
    return hist, bin_edges
