import torch
import data_processing as dp
import numpy as np
import interpolation as ip
import matplotlib.pyplot as plt
import interpolation as ip


def modify(source_data, output_data, interval):
    """
    source_data: (num_frames, 21, 3)
    """
    num_frames = len(source_data)
    exist_frames = list(range(0, num_frames, interval))

    for frame in exist_frames[1:-1]:
        output_data[frame] = (output_data[frame - 1] + output_data[frame + 1]) / 2.0

    diffs = np.zeros(source_data.shape)
    diffs[exist_frames] = output_data[exist_frames] - source_data[exist_frames]
    diffs = ip.linear_interpolate(diffs, interval)
    return output_data - diffs


def predict(eval_model, source_data, input_window, interval):
    num_sections = input_window // interval
    num_ref_sections = (num_sections - 1) // 2

    eval_model.eval()
    device = torch.device("cuda")
    lerp_data = ip.linear_interpolate(source_data, interval)
    output_data = lerp_data.copy()
    input_data, lerp_pos = dp.stay_at_origin(source_data)
    input_data, min_max_values = dp.bbox_fit(input_data, interval)
    num_frames, num_markers, num_axis = input_data.shape

    with torch.no_grad():
        for start in range(0, num_frames - input_window + 1, interval):
            end = start + input_window
            input_data_2d = input_data[start:end].reshape((input_window, -1))

            fitted_data_2d = dp.lost(input_data_2d, interval)

            fitted_data_3d = fitted_data_2d.reshape((input_window, 1, -1))

            data = torch.from_numpy(fitted_data_3d).double().to(device)
            fitted_output_3d = eval_model(data)
            fitted_output_3d = fitted_output_3d.cpu().detach().numpy()

            # 2d to 3d
            reverted_output = fitted_output_3d[:, 0].reshape((input_window, num_markers, num_axis))

            offset = interval * num_ref_sections

            if start == 0:
                output_data[start:end] = reverted_output
            else:
                output_data[start+offset:end] = reverted_output[offset:]

    output_data = modify(input_data, output_data, interval)

    output_data = dp.bbox_unfit(output_data, min_max_values)

    # align
    output_data = dp.align_bone_length(output_data, interval)

    # un stay
    output_data += lerp_pos
    return output_data
