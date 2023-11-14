import re

import numpy as np
# from matplotlib import pyplot as plt


# read data line by line and return a list
def extract_train_time(filepath):
    time_values = []
    with open(filepath, 'r') as file:
        for line in file:
            if "Train Time:" in line:
                time_ = line.strip().split(':')[-1].strip()
                time = time_[:-4].strip()
                time_values.append(float(time))
    return time_values


def extract_inference_metric(filepath):
    # report values in percentage
    f1s = []
    accs = []
    with open(filepath, 'r') as file:
        for line in file:
            f1_match = re.search(r'f1\s([-0-9.e]+|-)', line)
            acc_match = re.search(r'acc\s([-0-9.e]+|-)', line)

            if f1_match:
                f1_value = f1_match.group(1)
                f1s.append(float(f1_value))

            if acc_match:
                acc_value = acc_match.group(1)
                accs.append(float(acc_value))
    f1s = np.array(f1s).reshape((-1, 10))
    accs = np.array(accs).reshape((-1, 10))
    metrics = np.zeros(f1s.shape)
    metrics[:, :8] = f1s[:, :8]
    metrics[:, 8:] = accs[:, 8:]
    metrics = np.round(metrics * 100, 2)
    return metrics


def extract_inference_time(filepath, level='sample'):
    # the time will be reported in millisecond level
    time_values = []
    with open(filepath, 'r') as file:
        for line in file:
            if level == 'batch':
                pattern_str = 'The batch level inference time'
            elif level == 'sample':
                pattern_str = 'The sample level inference time'
            elif level == 'token':
                pattern_str = 'The token level inference time'
            else:
                raise ValueError('level must be one of batch, sample, token')
            if pattern_str in line:
                time = float(line.split(' ')[-1].strip()) * 1000
                time_values.append(time)
    time_values = np.array(time_values).reshape((-1, 10))
    time_values = np.round(time_values, 3)
    return time_values



def extract_transfer_metric(filepath):
    return extract_inference_metric(filepath)


if __name__ == '__main__':
    b_time_small = extract_inference_time('inference/inference-small-5xtime.txt', level='batch')
    b_time_base = extract_inference_time('inference/inference-base-5xtime.txt', level='batch')
    b_time_large = extract_inference_time('inference/inference-large-5xtime.txt', level='batch')
    b_time_small = b_time_small.reshape((5, -1, 10))
    b_time_base = b_time_base.reshape((5, -1, 10))
    b_time_large = b_time_large.reshape((5, -1, 10))
    b_time_small_mean = np.mean(b_time_small, axis=0)
    b_time_base_mean = np.mean(b_time_base, axis=0)
    b_time_large_mean = np.mean(b_time_large, axis=0)
    b_time_small_std = np.std(b_time_small, axis=0)
    b_time_base_std = np.std(b_time_base, axis=0)
    b_time_large_std = np.std(b_time_large, axis=0)

    s_time_small = extract_inference_time('inference/inference-small-5xtime.txt', level='sample')
    s_time_base = extract_inference_time('inference/inference-base-5xtime.txt', level='sample')
    s_time_large = extract_inference_time('inference/inference-large-5xtime.txt', level='sample')
    s_time_small = s_time_small.reshape((5, -1, 10))
    s_time_base = s_time_base.reshape((5, -1, 10))
    s_time_large = s_time_large.reshape((5, -1, 10))
    s_time_small_mean = np.mean(s_time_small, axis=0)
    s_time_base_mean = np.mean(s_time_base, axis=0)
    s_time_large_mean = np.mean(s_time_large, axis=0)
    s_time_small_std = np.std(s_time_small, axis=0)
    s_time_base_std = np.std(s_time_base, axis=0)
    s_time_large_std = np.std(s_time_large, axis=0)

    t_time_small = extract_inference_time('inference/inference-small-5xtime.txt', level='token')
    t_time_base = extract_inference_time('inference/inference-base-5xtime.txt', level='token')
    t_time_large = extract_inference_time('inference/inference-large-5xtime.txt', level='token')
    t_time_small = t_time_small.reshape((5, -1, 10))
    t_time_base = t_time_base.reshape((5, -1, 10))
    t_time_large = t_time_large.reshape((5, -1, 10))
    t_time_small_mean = np.mean(t_time_small, axis=0)
    t_time_base_mean = np.mean(t_time_base, axis=0)
    t_time_large_mean = np.mean(t_time_large, axis=0)
    t_time_small_std = np.std(t_time_small, axis=0)
    t_time_base_std = np.std(t_time_base, axis=0)
    t_time_large_std = np.std(t_time_large, axis=0)


    time_mean = np.concatenate((b_time_small_mean, b_time_base_mean, b_time_large_mean,
                                s_time_small_mean, s_time_base_mean, s_time_large_mean,
                                t_time_small_mean, t_time_base_mean, t_time_large_mean), axis=0)
    time_std = np.concatenate((b_time_small_std, b_time_base_std, b_time_large_std,
                               s_time_small_std, s_time_base_std, s_time_large_std,
                               t_time_small_std, t_time_base_std, t_time_large_std), axis=0)
    np.savetxt('inference_time_mean.csv', time_mean, delimiter='&')
    np.savetxt('inference_time_std.csv', time_std, delimiter='&')

    # for i in range(5):
    #     output = ''
    #     for j in range(10):
    #         # std_part = f'{time_std[i, j]:.1e}' if time_std[i, j] > 4e-4 else '0.0'
    #         mean_part = f'{time_mean[i, j]:.0f}'
    #         std_part = f'{time_std[i ,j]:5.2f}'
    #         output += '$' + mean_part + '_{' + std_part +'}$ & '
    #     output = output[:-2]
    #     print(output)


    # save as csv, using & as separator, keep 2 decimal
    # np.savetxt('inference_time.csv', time, delimiter='&', fmt='%.3f')
    # np.savetxt('inference_time.csv', time, fmt='%.3f')


