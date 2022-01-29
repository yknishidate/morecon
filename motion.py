import numpy as np


def replace_blank(array):
    return np.where(array == '', '0.0', array)


def get_exist_frames(data_1d):
    frames = np.where(data_1d != 0.0)[0]
    return frames


def extract_motion(csv_data, name):
    str_data = [row for row in csv_data if row[0] == name]
    str_array = np.array(str_data)
    str_array = replace_blank(str_array)
    float_array = str_array.astype(float)
    return float_array[:, 2:]


def reshape_to_3d(motion_2d):
    num_frames = motion_2d.shape[0]
    return np.reshape(motion_2d, (num_frames, 21, 3))


def reshape_to_2d(motion_3d):
    num_frames = motion_3d.shape[0]
    return np.reshape(motion_3d, (num_frames, 63))


class Motion:
    def __init__(self, csv_data, name) -> None:
        self.name = name
        self.data = reshape_to_3d(extract_motion(csv_data, name))

    def write_csv(self, writer):
        data_2d = reshape_to_2d(self.data).tolist()
        num_frames = self.data.shape[0]
        for frame in range(1, num_frames + 1):
            row = [self.name, str(frame)]
            row += data_2d[frame - 1]
            writer.writerow(row)
