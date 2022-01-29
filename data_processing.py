import numpy as np
import copy
import interpolation as ip


def stay_at_origin(data):
    first_pos = data[0, 0:1]
    last_pos = data[-1, 0:1]
    lerp_pos = np.linspace(first_pos, last_pos, len(data))
    return data - lerp_pos, lerp_pos


def bbox_fit(data_3d, interval):
    fitted_3d = copy.deepcopy(data_3d)
    exist_frames = list(range(0, len(data_3d), interval))
    mins = np.min(data_3d[exist_frames])
    maxs = np.max(data_3d[exist_frames])
    fitted_3d = (data_3d - mins) / (maxs - mins) * 2.0 - 1.0
    return fitted_3d, [mins, maxs]


def bbox_unfit(data_3d, min_max_values):
    unfitted_3d = copy.deepcopy(data_3d)
    mins, maxs = min_max_values
    unfitted_3d = (data_3d + 1.0) / 2.0 * (maxs - mins) + mins
    return unfitted_3d


def calc_loss(output, target):
    return np.mean(np.square(output - target))


def lost(marker_data, interval):
    output_data = copy.deepcopy(marker_data)
    for frame in range(len(marker_data)):
        if frame % interval != 0:
            output_data[frame, :] = 0.0
    return output_data


def get_value(data, index):
    index1 = np.floor(index).astype(int)
    index2 = np.ceil(index).astype(int)
    val1 = data[index1]
    val2 = data[index2]
    val = val1 + (val2 - val1) * (index - index1)
    return val


def resample(data, rate):
    new_data = []
    sample_count = int(len(data) // rate - 1)
    for i in range(sample_count):
        new_data.append(get_value(data, i * rate))
    return np.array(new_data)


PARENTS = [
    None,
    0, 1, 2, 3,
    2, 5, 6, 7,
    2, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19
]


def absolute_to_relative(data):
    relative_data = copy.deepcopy(data)
    for marker in range(1, 21):
        parent = PARENTS[marker]
        relative_data[:, marker] -= data[:, parent]
    return relative_data


def relative_to_absolute(data):
    absolute_data = copy.deepcopy(data)
    for marker in range(1, 21):
        parent = PARENTS[marker]
        absolute_data[:, marker] += absolute_data[:, parent]
    return absolute_data


def relative_to_length(data):
    length_data = np.zeros((data.shape[0], 21, 1))
    for frame in range(data.shape[0]):
        for marker in range(1, 21):
            length_data[frame, marker, 0] = np.linalg.norm(data[frame, marker])
    return length_data


def align_bone_length(data, interval):
    relative_data = absolute_to_relative(data)
    length_data = relative_to_length(relative_data)
    lerp_length_data = ip.linear_interpolate(length_data, interval)
    dir_data = np.zeros(relative_data.shape)
    dir_data[:, 1:] = relative_data[:, 1:] / length_data[:, 1:]
    for marker in range(1, 21):
        relative_data[:, marker] = lerp_length_data[:, marker] * dir_data[:, marker]
    absolute_data = relative_to_absolute(relative_data)
    return absolute_data
