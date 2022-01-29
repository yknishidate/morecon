import numpy as np
import copy

from interpolation import linear_interpolate


def stay_at_origin(data):
    first_pos = data[0, 0:1]
    last_pos = data[-1, 0:1]
    lerp_pos = np.linspace(first_pos, last_pos, len(data))
    return data - lerp_pos, lerp_pos


def invert_data(data, axis):
    cpy = copy.deepcopy(data)
    cpy[:, :, axis] = -cpy[:, :, axis]
    cpy[:, 5:9], cpy[:, 9:13] = cpy[:, 9:13].copy(), cpy[:, 5:9].copy()
    cpy[:, 13:17], cpy[:, 17:21] = cpy[:, 17:21].copy(), cpy[:, 13:17].copy()
    return cpy


def invert_time(data):
    return data[::-1]


def bbox_fit(data_3d, interval):
    fitted_3d = copy.deepcopy(data_3d)
    exist_frames = list(range(0, len(data_3d), interval))
    min_x = np.min(data_3d[exist_frames, :, 0])
    max_x = np.max(data_3d[exist_frames, :, 0])
    min_y = np.min(data_3d[exist_frames, :, 1])
    max_y = np.max(data_3d[exist_frames, :, 1])
    min_z = np.min(data_3d[exist_frames, :, 2])
    max_z = np.max(data_3d[exist_frames, :, 2])

    fitted_3d[:, :, 0] = (data_3d[:, :, 0] - min_x) / (max_x - min_x) * 2.0 - 1.0
    fitted_3d[:, :, 1] = (data_3d[:, :, 1] - min_y) / (max_y - min_y) * 2.0 - 1.0
    fitted_3d[:, :, 2] = (data_3d[:, :, 2] - min_z) / (max_z - min_z) * 2.0 - 1.0
    return fitted_3d, [min_x, max_x, min_y, max_y, min_z, max_z]


def bbox_unfit(data_3d, min_max_values):
    unfitted_3d = copy.deepcopy(data_3d)
    min_x, max_x, min_y, max_y, min_z, max_z = min_max_values
    unfitted_3d[:, :, 0] = (data_3d[:, :, 0] + 1.0) / 2.0 * (max_x - min_x) + min_x
    unfitted_3d[:, :, 1] = (data_3d[:, :, 1] + 1.0) / 2.0 * (max_y - min_y) + min_y
    unfitted_3d[:, :, 2] = (data_3d[:, :, 2] + 1.0) / 2.0 * (max_z - min_z) + min_z
    return unfitted_3d


def fit(data, interval=None):
    if interval is None:
        minval = np.min(data, axis=0)
        maxval = np.max(data, axis=0)
    else:
        exist_frames = list(range(0, len(data), interval))
        minval = np.min(data[exist_frames], axis=0)
        maxval = np.max(data[exist_frames], axis=0)

    midval = (minval + maxval) / 2.0
    scale = maxval - minval
    np.place(scale, scale < 1.0, 1.0)
    return (data - midval) / scale * 2.0, midval, scale


def unfit(data, midval, scale):
    return data / 2.0 * scale + midval


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


def absolute_to_relative(data):
    parents = [
        None,
        0, 1, 2, 3,
        2, 5, 6, 7,
        2, 9, 10, 11,
        0, 13, 14, 15,
        0, 17, 18, 19
    ]

    relative_data = copy.deepcopy(data)
    for marker in range(1, 21):
        parent = parents[marker]
        relative_data[:, marker] -= data[:, parent]
    return relative_data


def relative_to_absolute(data):
    parents = [
        None,
        0, 1, 2, 3,
        2, 5, 6, 7,
        2, 9, 10, 11,
        0, 13, 14, 15,
        0, 17, 18, 19
    ]

    absolute_data = copy.deepcopy(data)
    for marker in range(1, 21):
        parent = parents[marker]
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
    lerp_length_data = linear_interpolate(length_data, interval)
    dir_data = np.zeros(relative_data.shape)
    dir_data[:, 1:] = relative_data[:, 1:] / length_data[:, 1:]
    for marker in range(1, 21):
        relative_data[:, marker] = lerp_length_data[:, marker] * dir_data[:, marker]
    absolute_data = relative_to_absolute(relative_data)
    return absolute_data
