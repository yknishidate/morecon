import copy
import numpy as np
from scipy import interpolate


def linear_interpolate(data, interval):
    output = copy.deepcopy(data)
    num_frames, num_markers, num_axis = data.shape
    all_frames = list(range(num_frames))
    exist_frames = list(range(0, num_frames, interval))
    for marker in range(num_markers):
        for axis in range(num_axis):
            exist_x = exist_frames
            exist_y = data[exist_frames, marker, axis]
            interp_y = interpolate.interp1d(exist_x, exist_y)
            output[:, marker, axis] = np.array(interp_y(all_frames))
    return output


def quadratic_interpolate(data, interval):
    output = copy.deepcopy(data)
    num_frames, num_markers, num_axis = data.shape
    all_frames = list(range(num_frames))
    exist_frames = list(range(0, num_frames, interval))
    for marker in range(num_markers):
        for axis in range(num_axis):
            exist_x = exist_frames
            exist_y = data[exist_frames, marker, axis]
            interp_y = interpolate.interp1d(exist_x, exist_y, kind="quadratic")
            output[:, marker, axis] = np.array(interp_y(all_frames))
    return output
