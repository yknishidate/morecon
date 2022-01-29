import torch
import numpy as np
import random
import copy
import data_processing as dp


def make_dataset(motions, interval, input_window, rates=[1.0]):
    """
    returns: (data_size, window, feature)
    """
    device = torch.device("cuda")
    source = []
    target = []
    for motion in motions:
        data, _ = dp.stay_at_origin(motion.data)
        for rate in rates:
            resampled_data = dp.resample(data, rate)
            resampled_data = dp.bbox_fit(resampled_data, interval)[0]
            num_frames = len(resampled_data)
            num_windows = num_frames - input_window + 1
            for start in range(0, num_windows, 3):
                end = start + input_window
                data_2d = resampled_data[start:end].reshape((input_window, -1))
                target.append(data_2d)
                source.append(dp.lost(data_2d, interval))
    source = torch.from_numpy(np.array(source)).double()
    target = torch.from_numpy(np.array(target)).double()
    print("data:", source.shape)
    return source.to(device), target.to(device)


def make_predict_data(motion, interval):
    num_frames = len(motion.data)
    exist_frames = list(range(0, num_frames, interval))
    target_data = motion.data[:exist_frames[-1] + 1]
    return copy.deepcopy(target_data), target_data


def get_both_batch(source, target, batch_size, input_window):
    indices = random.sample(range(len(source)), batch_size)
    batch_source = source[indices]
    batch_target = target[indices]
    batch_source = torch.permute(batch_source, (1, 0, 2))
    batch_source = torch.reshape(batch_source, (input_window, batch_size, -1))
    batch_target = torch.permute(batch_target, (1, 0, 2))
    batch_target = torch.reshape(batch_target, (input_window, batch_size, -1))
    return batch_source, batch_target


def get_train_valid(motions, min_valid_size=2):
    valid_size = max(int(len(motions) * 0.15), min_valid_size)
    train_motions = motions[:-valid_size]
    valid_motions = motions[-valid_size:]
    print("train_motions:", len(train_motions))
    print("valid_motions:", len(valid_motions))
    return train_motions, valid_motions
