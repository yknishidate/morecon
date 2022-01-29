import torch
import data_processing as dp
import numpy as np
import interpolation as ip
import copy
import dataset


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
    num_frames = input_data.shape[0]

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
            reverted_output = fitted_output_3d[:, 0].reshape((input_window, 21, 3))

            offset = interval * num_ref_sections

            if start == 0:
                output_data[start:end] = reverted_output
            else:
                output_data[start+offset:end] = reverted_output[offset:]

    output_data = modify(input_data, output_data, interval)
    output_data = dp.bbox_unfit(output_data, min_max_values)
    output_data = dp.align_bone_length(output_data, interval)
    output_data += lerp_pos
    return output_data


def search_best_weights(model, test_motions, name, interval, input_window):
    model0 = copy.deepcopy(model)
    model1 = copy.deepcopy(model)
    model2 = copy.deepcopy(model)
    model0.load_state_dict(torch.load(f"model/{name}_0.pth"))
    model1.load_state_dict(torch.load(f"model/{name}_1.pth"))
    model2.load_state_dict(torch.load(f"model/{name}_2.pth"))

    # all motions
    pred_data0 = []
    pred_data1 = []
    pred_data2 = []
    lerp_data = []
    for test_motion in test_motions:
        if test_motion.data.shape[0] < input_window:
            pred_data0.append(0.0)
            pred_data1.append(0.0)
            pred_data2.append(0.0)
            lerp_data.append(0.0)
            continue
        source_data, target_data = dataset.make_predict_data(test_motion, interval)
        source_data = dp.lost(source_data, interval)

        pred_data0.append(predict(model0, source_data, input_window, interval))
        pred_data1.append(predict(model1, source_data, input_window, interval))
        pred_data2.append(predict(model2, source_data, input_window, interval))
        lerp_data.append(ip.linear_interpolate(source_data, interval))

    best_weights = np.zeros((21, 4))
    min_loss = np.full((21), float("inf"))
    for marker in range(21):
        for w0 in range(0, 11):
            for w1 in range(0, 11 - w0):
                for w2 in range(0, 11 - w0 - w1):
                    w3 = 10 - w0 - w1 - w2
                    pred_loss = 0
                    for index, test_motion in enumerate(test_motions):
                        if test_motion.data.shape[0] < input_window:
                            continue
                        source_data, target_data = dataset.make_predict_data(test_motion, interval)
                        data0 = pred_data0[index][:, marker]
                        data1 = pred_data1[index][:, marker]
                        data2 = pred_data2[index][:, marker]
                        data3 = lerp_data[index][:, marker]
                        pred_data = (data0 * w0 + data1 * w1 + data2 * w2 + data3 * w3) / 10.0
                        pred_loss += dp.calc_loss(pred_data, target_data[:, marker])
                    if pred_loss < min_loss[marker]:
                        min_loss[marker] = pred_loss
                        best_weights[marker] = [w0, w1, w2, w3]
    return best_weights


def predict_with_weights(model, source_data, weights, name, interval, input_window):
    model0 = copy.deepcopy(model)
    model1 = copy.deepcopy(model)
    model2 = copy.deepcopy(model)
    model0.load_state_dict(torch.load(f"model/{name}_0.pth"))
    model1.load_state_dict(torch.load(f"model/{name}_1.pth"))
    model2.load_state_dict(torch.load(f"model/{name}_2.pth"))

    pred_data0 = predict(model0, source_data, input_window, interval)
    pred_data1 = predict(model1, source_data, input_window, interval)
    pred_data2 = predict(model2, source_data, input_window, interval)
    lerp_data = ip.linear_interpolate(source_data, interval)

    pred_data = np.zeros(source_data.shape)
    for marker in range(21):
        w0, w1, w2, w3 = weights[marker]
        pred_data[:, marker] = (pred_data0[:, marker] * w0 +
                                pred_data1[:, marker] * w1 +
                                pred_data2[:, marker] * w2 +
                                lerp_data[:, marker] * w3) / 10.0
    return pred_data
