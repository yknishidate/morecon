import copy
import numpy as np
import scipy.interpolate


def interpolate(data, interval, method):
    output = copy.deepcopy(data)
    num_frames, num_markers, num_axis = data.shape
    all_frames = list(range(num_frames))
    exist_frames = list(range(0, num_frames, interval))
    for marker in range(num_markers):
        for axis in range(num_axis):
            exist_x = exist_frames
            exist_y = data[exist_frames, marker, axis]
            interp_y = scipy.interpolate.interp1d(exist_x, exist_y, method)
            output[:, marker, axis] = np.array(interp_y(all_frames))
    return output


def linear_interpolate(data, interval):
    return interpolate(data, interval, "linear")


def quadratic_interpolate(data, interval):
    return interpolate(data, interval, "quadratic")
