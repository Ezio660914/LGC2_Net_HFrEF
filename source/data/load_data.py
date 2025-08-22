# -*- coding: utf-8 -*-
import csv
import os
import wave

import numpy as np
from scipy import signal
from wfdb import processing


def read_wav_data(filename):
    '''
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    '''
    # print('filename=', filename)
    wav = wave.open(filename, "rb")  # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes()  # 获取帧数
    # print('num_frame=',num_frame)
    num_channel = wav.getnchannels()  # 获取声道数
    # print('num_channel=',num_channel)
    framerate = wav.getframerate()  # 获取帧速率
    # print('framerate=',framerate)
    num_sample_width = wav.getsampwidth()  # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame)  # 读取全部的帧
    wav.close()  # 关闭流
    # wave_data = np.fromstring(str_data, dtype = np.short) # 将声音文件数据转换为数组矩阵形式
    wave_data = np.fromstring(str_data, dtype=np.int16)
    # wave_data = wave_data*1.0/(max(abs(wave_data)))#wave幅值归一化
    # print(wave_data.shape)
    wave_data.shape = -1, num_channel  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T  # 将矩阵转置
    return wave_data, framerate


def band_pass_filter(original_signal, order, fc1, fc2, fs):
    '''
    中值滤波器
    :param original_signal: 音频数据
    :param order: 滤波器阶数
    :param fc1: 截止频率
    :param fc2: 截止频率
    :param fs: 音频采样率
    :return: 滤波后的音频数据
    '''
    b, a = signal.butter(N=order, Wn=[2 * fc1 / fs, 2 * fc2 / fs], btype='bandpass')
    # b, a = signal.butter(N=order, Wn=2*fc1/fs, btype='highpass')
    new_signal = signal.lfilter(b, a, original_signal)
    return new_signal


def notch_filter(original_signal, w0, fs):
    '''
    陷波滤波器
    :param original_signal: 音频数据
    :w0: 去除的频率
    :Q: 品质因子
    :fs: 采样频率
    :return: 滤波后的音频数据
    '''
    Q = 30
    b, a = signal.iirnotch(w0, Q, fs)
    new_signal = signal.lfilter(b, a, original_signal)
    return new_signal


def high_pass_filter(original_signal, order, fc1, fs):
    '''
    中值滤波器
    :param original_signal: 音频数据
    :param order: 滤波器阶数
    :param fc1: 截止频率
    :param fs: 音频采样率
    :return: 滤波后的音频数据
    '''
    b, a = signal.butter(N=order, Wn=2 * fc1 / fs, btype='highpass')
    new_signal = signal.lfilter(b, a, original_signal)
    return new_signal


def lower_pass_filter(original_signal, order, fc1, fs):
    '''
    中值滤波器
    :param original_signal: 音频数据
    :param order: 滤波器阶数
    :param fc1: 截止频率
    :param fs: 音频采样率
    :return: 滤波后的音频数据
    '''
    b, a = signal.butter(N=order, Wn=2 * fc1 / fs, btype='lowpass')
    new_signal = signal.lfilter(b, a, original_signal)
    return new_signal


def enframe(signal, nw, inc):
    '''
    将音频信号转化为帧。

    参数含义：

    signal:原始音频型号

    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)

    inc:相邻帧的间隔（同上定义）

    '''
    # 不足部分舍弃
    signal_length = len(signal)  # 信号总长度
    # print('signal_length=',signal_length)

    if signal_length <= nw:  # 若信号长度小于一个帧的长度，则帧数定义为1

        nf = 1

    else:  # 否则，计算帧的总长度

        nf = int(np.floor((1.0 * signal_length - nw + inc) / inc))
        pad_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的铺平后的长度
        indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
        # print('indices=',indices)
        indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
        frames = signal[indices]  # 得到帧信号
        # win=np.tile(winfunc,(nf,1)) #window窗函数，这里默认取1
        # return frames*win  #返回帧信号矩阵
        return frames


def Normalized(record):
    x = record[:, 0]
    x = (x - np.mean(x)) / (np.std(x))
    x = x.reshape(-1, 1)
    x = x.T
    return x


def Normalized_PCG(record):
    x = record[0, :]
    x = (x - np.mean(x)) / (np.std(x))
    x = x.reshape(-1, 1)
    x = x.T
    return x


def load_data_cross_validate_train_test_four_locations_parallel_in_house_data(data_dir, k_folds, current_fold, resample_frequency, EF_threshold):  # 同时读取四个采集部位信号

    ECG_train_val_pos_record_list = []
    ECG_train_val_pos_record_name_list = []
    ECG_train_val_neg_record_list = []
    ECG_train_val_neg_record_name_list = []
    ECG_test_pos_record_list = []
    ECG_test_pos_record_name_list = []
    ECG_test_neg_record_list = []
    ECG_test_neg_record_name_list = []

    PCG_train_val_pos_record_list = []
    PCG_train_val_pos_record_name_list = []
    PCG_train_val_neg_record_list = []
    PCG_train_val_neg_record_name_list = []
    PCG_test_pos_record_list = []
    PCG_test_pos_record_name_list = []
    PCG_test_neg_record_list = []
    PCG_test_neg_record_name_list = []

    location = ['APEX', 'LLSB', 'LUSB', 'RUSB']

    for i in range(k_folds):

        if i != current_fold:
            print('i=', i)
            fold_path = os.path.join(data_dir, f'fold_{i}')
            # 读取recording名字和标签
            patient_name_list = []
            label_content = []  # 用来存储标签
            csv_file = csv.reader(open(os.path.join(data_dir, f'fold_{i}_label.csv'), 'r'))
            for line in csv_file:
                if line[0].split('_')[0] not in patient_name_list:
                    patient_name_list.append(line[0].split('_')[0])
                    label_content.append(line[1])

            # 数据预处理:归一化，去噪
            for j in range(len(patient_name_list)):
                ECG_patinet_reocrdings_four_locations_list = []
                PCG_patinet_reocrdings_four_locations_list = []
                for k in range(len(location)):
                    ECG_wave_data, ECG_fs = read_wav_data(os.path.join(fold_path, patient_name_list[j] + '_' + location[k] + '_ECG.wav'))  # shape(1,n)
                    ECG_recording_resample, _ = processing.resample_sig(x=ECG_wave_data[0, :], fs=ECG_fs, fs_target=resample_frequency)  # shape(n,)
                    ECG_recording_alignment = (ECG_recording_resample[100:]).reshape(-1, 1)  # shape(n-100,1)
                    ECG_recording_Normalized = Normalized(ECG_recording_alignment)  # shape(1,n-100)
                    ECG_recording_after_filtering = ECG_recording_Normalized  # ECG信号未进行滤波，shape(1,n-100)
                    ECG_patinet_reocrdings_four_locations_list.append(ECG_recording_after_filtering)

                    PCG_wave_data, PCG_fs = read_wav_data(os.path.join(fold_path, patient_name_list[j] + '_' + location[k] + '_PCG.wav'))  # shape(1,n)
                    PCG_recording_resample, _ = processing.resample_sig(x=PCG_wave_data[0, :], fs=PCG_fs, fs_target=resample_frequency)  # shape(n,)
                    PCG_recording_alignment = (PCG_recording_resample[100:]).reshape(-1, 1)  # shape(n-100,1)
                    PCG_recording_Normalized = Normalized(PCG_recording_alignment)  # shape(1,n-100)
                    PCG_recording_after_filtering = band_pass_filter(PCG_recording_Normalized, 5, 25, 400, resample_frequency)  # (1,n-100)五阶巴特沃夫高通滤波器滤波
                    PCG_patinet_reocrdings_four_locations_list.append(PCG_recording_after_filtering)

                if float(label_content[j]) <= EF_threshold:
                    ECG_train_val_pos_record_list.append(ECG_patinet_reocrdings_four_locations_list)
                    ECG_train_val_pos_record_name_list.append(patient_name_list[j])
                    PCG_train_val_pos_record_list.append(PCG_patinet_reocrdings_four_locations_list)
                    PCG_train_val_pos_record_name_list.append(patient_name_list[j])
                if float(label_content[j]) > EF_threshold:
                    ECG_train_val_neg_record_list.append(ECG_patinet_reocrdings_four_locations_list)
                    ECG_train_val_neg_record_name_list.append(patient_name_list[j])
                    PCG_train_val_neg_record_list.append(PCG_patinet_reocrdings_four_locations_list)
                    PCG_train_val_neg_record_name_list.append(patient_name_list[j])

        # 读取测试集
        if i == current_fold:
            print('i=test')
            fold_path = os.path.join(data_dir, f'fold_{i}')
            # 读取recording名字和标签
            patient_name_list = []
            label_content = []  # 用来存储标签
            csv_file = csv.reader(open(os.path.join(data_dir, f'fold_{i}_label.csv'), 'r'))
            for line in csv_file:
                if line[0].split('_')[0] not in patient_name_list:
                    patient_name_list.append(line[0].split('_')[0])
                    label_content.append(line[1])

            # 数据预处理:归一化，去噪
            for j in range(len(patient_name_list)):
                ECG_patinet_reocrdings_four_locations_list = []
                PCG_patinet_reocrdings_four_locations_list = []
                for k in range(len(location)):
                    ECG_wave_data, ECG_fs = read_wav_data(os.path.join(fold_path, patient_name_list[j] + '_' + location[k] + '_ECG.wav'))  # shape(1,n)
                    ECG_recording_resample, _ = processing.resample_sig(x=ECG_wave_data[0, :], fs=ECG_fs, fs_target=resample_frequency)  # shape(n,)
                    ECG_recording_alignment = (ECG_recording_resample[100:]).reshape(-1, 1)  # shape(n-100,1)
                    ECG_recording_Normalized = Normalized(ECG_recording_alignment)  # shape(1,n-100)
                    ECG_recording_after_filtering = ECG_recording_Normalized  # ECG信号未进行滤波，shape(1,n-100)
                    ECG_patinet_reocrdings_four_locations_list.append(ECG_recording_after_filtering)

                    PCG_wave_data, PCG_fs = read_wav_data(os.path.join(fold_path, patient_name_list[j] + '_' + location[k] + '_PCG.wav'))  # shape(1,n)
                    PCG_recording_resample, _ = processing.resample_sig(x=PCG_wave_data[0, :], fs=PCG_fs, fs_target=resample_frequency)  # shape(n,)
                    PCG_recording_alignment = (PCG_recording_resample[100:]).reshape(-1, 1)  # shape(n-100,1)
                    PCG_recording_Normalized = Normalized(PCG_recording_alignment)  # shape(1,n-100)
                    PCG_recording_after_filtering = band_pass_filter(PCG_recording_Normalized, 5, 25, 400, resample_frequency)  # (1,n-100)五阶巴特沃夫高通滤波器滤波
                    PCG_patinet_reocrdings_four_locations_list.append(PCG_recording_after_filtering)
                if float(label_content[j]) <= EF_threshold:
                    ECG_test_pos_record_list.append(ECG_patinet_reocrdings_four_locations_list)
                    ECG_test_pos_record_name_list.append(patient_name_list[j])
                    PCG_test_pos_record_list.append(PCG_patinet_reocrdings_four_locations_list)
                    PCG_test_pos_record_name_list.append(patient_name_list[j])
                if float(label_content[j]) > EF_threshold:
                    ECG_test_neg_record_list.append(ECG_patinet_reocrdings_four_locations_list)
                    ECG_test_neg_record_name_list.append(patient_name_list[j])
                    PCG_test_neg_record_list.append(PCG_patinet_reocrdings_four_locations_list)
                    PCG_test_neg_record_name_list.append(patient_name_list[j])

    print('ECG_train_val_pos_record_name_list=', len(ECG_train_val_pos_record_name_list))
    print('ECG_train_val_pos_record_name_list=', ECG_train_val_pos_record_name_list)

    # 各子集阳性、阴性样本合并并打乱
    # ECG信号
    ECG_train_record_list = ECG_train_val_neg_record_list + ECG_train_val_pos_record_list
    ECG_train_name_list = ECG_train_val_neg_record_name_list + ECG_train_val_pos_record_name_list
    ECG_train_label_list = list(np.zeros(shape=(len(ECG_train_val_neg_record_list),))) + list(np.ones(shape=(len(ECG_train_val_pos_record_list),)))
    ECG_train_name_record_label_list = list(zip(ECG_train_name_list, ECG_train_record_list, ECG_train_label_list))
    np.random.seed(1234)
    np.random.shuffle(ECG_train_name_record_label_list)
    ECG_train_name_list, ECG_train_record_list, ECG_train_label_list = zip(*ECG_train_name_record_label_list)

    ECG_test_record_list = ECG_test_neg_record_list + ECG_test_pos_record_list
    ECG_test_name_list = ECG_test_neg_record_name_list + ECG_test_pos_record_name_list
    ECG_test_label_list = list(np.zeros(shape=(len(ECG_test_neg_record_list),))) + list(np.ones(shape=(len(ECG_test_pos_record_list),)))
    ECG_test_name_record_label_list = list(zip(ECG_test_name_list, ECG_test_record_list, ECG_test_label_list))
    np.random.seed(3456)
    np.random.shuffle(ECG_test_name_record_label_list)
    ECG_test_name_list, ECG_test_record_list, ECG_test_label_list = zip(*ECG_test_name_record_label_list)

    print('ECG_train:', len(ECG_train_name_list), len(ECG_train_record_list), len(ECG_train_label_list))
    print('ECG_test:', len(ECG_test_name_list), len(ECG_test_record_list), len(ECG_test_label_list))

    # PCG信号
    PCG_train_record_list = PCG_train_val_neg_record_list + PCG_train_val_pos_record_list
    PCG_train_name_list = PCG_train_val_neg_record_name_list + PCG_train_val_pos_record_name_list
    PCG_train_label_list = list(np.zeros(shape=(len(PCG_train_val_neg_record_list),))) + list(np.ones(shape=(len(PCG_train_val_pos_record_list),)))
    PCG_train_name_record_label_list = list(zip(PCG_train_name_list, PCG_train_record_list, PCG_train_label_list))
    np.random.seed(1234)  # PCG与ECG的打乱顺序必须一致
    np.random.shuffle(PCG_train_name_record_label_list)
    PCG_train_name_list, PCG_train_record_list, PCG_train_label_list = zip(*PCG_train_name_record_label_list)

    PCG_test_record_list = PCG_test_neg_record_list + PCG_test_pos_record_list
    PCG_test_name_list = PCG_test_neg_record_name_list + PCG_test_pos_record_name_list
    PCG_test_label_list = list(np.zeros(shape=(len(PCG_test_neg_record_list),))) + list(np.ones(shape=(len(PCG_test_pos_record_list),)))
    PCG_test_name_record_label_list = list(zip(PCG_test_name_list, PCG_test_record_list, PCG_test_label_list))
    np.random.seed(3456)
    np.random.shuffle(PCG_test_name_record_label_list)
    PCG_test_name_list, PCG_test_record_list, PCG_test_label_list = zip(*PCG_test_name_record_label_list)

    print('PCG_train:', len(PCG_train_name_list), len(PCG_train_record_list), len(PCG_train_label_list))
    print('PCG_test:', len(PCG_test_name_list), len(PCG_test_record_list), len(PCG_test_label_list))

    # record分段 ECG信号
    ECG_train_patch_list = []
    ECG_train_patch_label_list = []
    ECG_train_patch_name_list = []
    ECG_train_num_list = []
    for i in range(len(ECG_train_record_list)):
        ECG_train_record = ECG_train_record_list[i]
        Frame_APEX = enframe(ECG_train_record[0][0, :], 2560, 1280)
        Frame_LLSB = enframe(ECG_train_record[1][0, :], 2560, 1280)
        Frame_LUSB = enframe(ECG_train_record[2][0, :], 2560, 1280)
        Frame_RUSB = enframe(ECG_train_record[3][0, :], 2560, 1280)
        Frame_APEX = Frame_APEX[..., np.newaxis]
        Frame_LLSB = Frame_LLSB[..., np.newaxis]
        Frame_LUSB = Frame_LUSB[..., np.newaxis]
        Frame_RUSB = Frame_RUSB[..., np.newaxis]  # (n,2560,1)

        Frame = np.concatenate((Frame_APEX, Frame_LLSB, Frame_LUSB, Frame_RUSB), axis=2)  # (n,2560,4)

        num = Frame.shape[0]
        for j in range(0, num, 1):
            ECG_train_patch = Frame[j, :, :]  # (2560,4)
            ECG_train_patch = ECG_train_patch[np.newaxis, ...]  # (1,2560,4)
            ECG_train_patch_list.append(ECG_train_patch)
            ECG_train_patch_label_list.append(ECG_train_label_list[i])
            ECG_train_patch_name_list.append(ECG_train_name_list[i])
        ECG_train_num_list.append(num)

    # 统计训练数据中阳性、阴性Patch的数量
    ECG_train_pos_patch_num = 0
    ECG_train_neg_patch_num = 0
    for i in range(len(ECG_train_patch_label_list)):
        if ECG_train_patch_label_list[i] == 0:
            ECG_train_neg_patch_num = ECG_train_neg_patch_num + 1
        if ECG_train_patch_label_list[i] == 1:
            ECG_train_pos_patch_num = ECG_train_pos_patch_num + 1
    print('ECG_train_pos_patch_num=', ECG_train_pos_patch_num)
    print('ECG_train_neg_patch_num=', ECG_train_neg_patch_num)

    ECG_test_patch_list = []
    ECG_test_patch_label_list = []
    ECG_test_patch_name_list = []
    ECG_test_num_list = []
    for i in range(len(ECG_test_record_list)):
        ECG_test_record = ECG_test_record_list[i]
        Frame_APEX = enframe(ECG_test_record[0][0, :], 2560, 1280)
        Frame_LLSB = enframe(ECG_test_record[1][0, :], 2560, 1280)
        Frame_LUSB = enframe(ECG_test_record[2][0, :], 2560, 1280)
        Frame_RUSB = enframe(ECG_test_record[3][0, :], 2560, 1280)
        Frame_APEX = Frame_APEX[..., np.newaxis]
        Frame_LLSB = Frame_LLSB[..., np.newaxis]
        Frame_LUSB = Frame_LUSB[..., np.newaxis]
        Frame_RUSB = Frame_RUSB[..., np.newaxis]  # (n,2560,1)

        Frame = np.concatenate((Frame_APEX, Frame_LLSB, Frame_LUSB, Frame_RUSB), axis=2)  # (n,2560,4)

        num = Frame.shape[0]
        for j in range(0, num, 1):
            ECG_test_patch = Frame[j, :, :]  # (2560,4)
            ECG_test_patch = ECG_test_patch[np.newaxis, ...]  # (1,2560,4)
            ECG_test_patch_list.append(ECG_test_patch)
            ECG_test_patch_label_list.append(ECG_test_label_list[i])
            ECG_test_patch_name_list.append(ECG_test_name_list[i])
        ECG_test_num_list.append(num)

    ECG_train_x = np.vstack(ECG_train_patch_list)  # shape(n,2560,4)
    ECG_train_y = np.vstack(ECG_train_patch_label_list)  # shape(n,1)
    ECG_test_x = np.vstack(ECG_test_patch_list)  # (n,2560,4)
    ECG_test_y = np.vstack(ECG_test_patch_label_list)  # shape(n,1)
    ECG_test_record_label = np.vstack(ECG_test_label_list)

    # record分段 PCG信号
    PCG_train_patch_list = []
    PCG_train_patch_label_list = []
    PCG_train_patch_name_list = []
    PCG_train_num_list = []
    for i in range(len(PCG_train_record_list)):
        PCG_train_record = PCG_train_record_list[i]
        Frame_APEX = enframe(PCG_train_record[0][0, :], 2560, 1280)
        Frame_LLSB = enframe(PCG_train_record[1][0, :], 2560, 1280)
        Frame_LUSB = enframe(PCG_train_record[2][0, :], 2560, 1280)
        Frame_RUSB = enframe(PCG_train_record[3][0, :], 2560, 1280)
        Frame_APEX = Frame_APEX[..., np.newaxis]
        Frame_LLSB = Frame_LLSB[..., np.newaxis]
        Frame_LUSB = Frame_LUSB[..., np.newaxis]
        Frame_RUSB = Frame_RUSB[..., np.newaxis]  # (n,2560,1)

        Frame = np.concatenate((Frame_APEX, Frame_LLSB, Frame_LUSB, Frame_RUSB), axis=2)  # (n,2560,4)

        num = Frame.shape[0]
        for j in range(0, num, 1):
            PCG_train_patch = Frame[j, :, :]
            PCG_train_patch = PCG_train_patch[np.newaxis, ...]  # (1,2560,4)
            PCG_train_patch_list.append(PCG_train_patch)
            PCG_train_patch_label_list.append(PCG_train_label_list[i])
            PCG_train_patch_name_list.append(PCG_train_name_list[i])
        PCG_train_num_list.append(num)

    # 统计训练数据中阳性、阴性Patch的数量
    PCG_train_pos_patch_num = 0
    PCG_train_neg_patch_num = 0
    for i in range(len(PCG_train_patch_label_list)):
        if PCG_train_patch_label_list[i] == 0:
            PCG_train_neg_patch_num = PCG_train_neg_patch_num + 1
        if PCG_train_patch_label_list[i] == 1:
            PCG_train_pos_patch_num = PCG_train_pos_patch_num + 1
    print('PCG_train_pos_patch_num=', PCG_train_pos_patch_num)
    print('PCG_train_neg_patch_num=', PCG_train_neg_patch_num)

    PCG_test_patch_list = []
    PCG_test_patch_label_list = []
    PCG_test_patch_name_list = []
    PCG_test_num_list = []
    for i in range(len(PCG_test_record_list)):
        PCG_test_record = PCG_test_record_list[i]
        Frame_APEX = enframe(PCG_test_record[0][0, :], 2560, 1280)
        Frame_LLSB = enframe(PCG_test_record[1][0, :], 2560, 1280)
        Frame_LUSB = enframe(PCG_test_record[2][0, :], 2560, 1280)
        Frame_RUSB = enframe(PCG_test_record[3][0, :], 2560, 1280)
        Frame_APEX = Frame_APEX[..., np.newaxis]
        Frame_LLSB = Frame_LLSB[..., np.newaxis]
        Frame_LUSB = Frame_LUSB[..., np.newaxis]
        Frame_RUSB = Frame_RUSB[..., np.newaxis]  # (n,2560,1)

        Frame = np.concatenate((Frame_APEX, Frame_LLSB, Frame_LUSB, Frame_RUSB), axis=2)  # (n,2560,4)

        num = Frame.shape[0]
        for j in range(0, num, 1):
            PCG_test_patch = Frame[j, :, :]
            PCG_test_patch = PCG_test_patch[np.newaxis, ...]  # (1,2560,4)
            PCG_test_patch_list.append(PCG_test_patch)
            PCG_test_patch_label_list.append(PCG_test_label_list[i])
            PCG_test_patch_name_list.append(PCG_test_name_list[i])
        PCG_test_num_list.append(num)

    PCG_train_x = np.vstack(PCG_train_patch_list)  # shape(n,2560,4)
    PCG_train_y = np.vstack(PCG_train_patch_label_list)  # shape(n,1)
    PCG_test_x = np.vstack(PCG_test_patch_list)  # shape(n,2560,4)
    PCG_test_y = np.vstack(PCG_test_patch_label_list)  # shape(n,1)
    PCG_test_record_label = np.vstack(PCG_test_label_list)

    ECG_train_x = ECG_train_x[..., np.newaxis]  # shape(n,2560,4,1)
    PCG_train_x = PCG_train_x[..., np.newaxis]  # shape(n,2560,4,1)
    ECG_PCG_train_x = np.concatenate((ECG_train_x, PCG_train_x), axis=3)  # shape(n,2560,4,2)
    ECG_test_x = ECG_test_x[..., np.newaxis]  # shape(n,2560,4,1)
    PCG_test_x = PCG_test_x[..., np.newaxis]  # shape(n,2560,4,1)
    ECG_PCG_test_x = np.concatenate((ECG_test_x, PCG_test_x), axis=3)  # shape(n,2560,4,2)

    if ((ECG_train_y.all() == PCG_train_y.all()) and (ECG_test_y.all() == PCG_test_y.all()) and (ECG_test_record_label.all() == PCG_test_record_label.all()) and \
            (ECG_test_num_list == PCG_test_num_list) and (ECG_test_patch_name_list == PCG_test_patch_name_list)):
        print('all equal')

    return ECG_PCG_train_x, ECG_train_y, ECG_PCG_test_x, ECG_test_y, ECG_test_record_label, ECG_test_num_list, ECG_test_patch_name_list, ECG_test_name_list


def load_data_cross_validate_train_test_four_locations_parallel_external_validation_data(data_dir, resample_frequency, EF_threshold):  # 同时读取四个采集部位信号

    ECG_test_pos_record_list = []
    ECG_test_pos_record_name_list = []
    ECG_test_neg_record_list = []
    ECG_test_neg_record_name_list = []

    PCG_test_pos_record_list = []
    PCG_test_pos_record_name_list = []
    PCG_test_neg_record_list = []
    PCG_test_neg_record_name_list = []

    location = ['APEX', 'LLSB', 'LUSB', 'RUSB']

    # 读取测试集
    data_path = os.path.join(data_dir, 'test_set')
    # 读取recording名字和标签
    patient_name_list = []
    label_content = []  # 用来存储标签
    csv_file = csv.reader(open(os.path.join(data_dir, 'test_set_label.csv'), 'r'))
    for line in csv_file:
        patient_name_list.append(str(line[0]))  # 列表元素类型由数值转化为字符串
        label_content.append(line[1])

    # 数据预处理:归一化，去噪
    for j in range(len(patient_name_list)):
        ECG_patinet_reocrdings_four_locations_list = []
        PCG_patinet_reocrdings_four_locations_list = []
        for k in range(len(location)):
            ECG_wave_data, ECG_fs = read_wav_data(os.path.join(data_path, patient_name_list[j] + '_' + location[k] + '_ECG.wav'))  # shape(1,n)
            ECG_recording_resample, _ = processing.resample_sig(x=ECG_wave_data[0, :], fs=ECG_fs, fs_target=resample_frequency)  # shape(n,)
            ECG_recording_alignment = (ECG_recording_resample[100:]).reshape(-1, 1)  # shape(n-100,1)
            ECG_recording_Normalized = Normalized(ECG_recording_alignment)  # shape(1,n-100)
            ECG_recording_after_filtering = ECG_recording_Normalized  # ECG信号未进行滤波，shape(1,n-100)
            ECG_patinet_reocrdings_four_locations_list.append(ECG_recording_after_filtering)

            PCG_wave_data, PCG_fs = read_wav_data(os.path.join(data_path, patient_name_list[j] + '_' + location[k] + '_PCG.wav'))  # shape(1,n)
            PCG_recording_resample, _ = processing.resample_sig(x=PCG_wave_data[0, :], fs=PCG_fs, fs_target=resample_frequency)  # shape(n,)
            PCG_recording_alignment = (PCG_recording_resample[100:]).reshape(-1, 1)  # shape(n-100,1)
            PCG_recording_Normalized = Normalized(PCG_recording_alignment)  # shape(1,n-100)
            PCG_recording_after_filtering = band_pass_filter(PCG_recording_Normalized, 5, 25, 400, resample_frequency)  # (1,n-100)五阶巴特沃夫高通滤波器滤波
            PCG_patinet_reocrdings_four_locations_list.append(PCG_recording_after_filtering)

        if float(label_content[j]) <= EF_threshold:
            ECG_test_pos_record_list.append(ECG_patinet_reocrdings_four_locations_list)
            ECG_test_pos_record_name_list.append(patient_name_list[j])
            PCG_test_pos_record_list.append(PCG_patinet_reocrdings_four_locations_list)
            PCG_test_pos_record_name_list.append(patient_name_list[j])
        if float(label_content[j]) > EF_threshold:
            ECG_test_neg_record_list.append(ECG_patinet_reocrdings_four_locations_list)
            ECG_test_neg_record_name_list.append(patient_name_list[j])
            PCG_test_neg_record_list.append(PCG_patinet_reocrdings_four_locations_list)
            PCG_test_neg_record_name_list.append(patient_name_list[j])

    print('ECG_test_pos_record_name_list=', len(ECG_test_pos_record_name_list))
    print('ECG_test_pos_record_name_list=', ECG_test_pos_record_name_list)
    print('ECG_test_neg_record_name_list=', len(ECG_test_neg_record_name_list))
    print('ECG_test_neg_record_name_list=', ECG_test_neg_record_name_list)

    # 各子集阳性、阴性样本合并并打乱
    # ECG信号
    ECG_test_record_list = ECG_test_neg_record_list + ECG_test_pos_record_list
    ECG_test_name_list = ECG_test_neg_record_name_list + ECG_test_pos_record_name_list
    ECG_test_label_list = list(np.zeros(shape=(len(ECG_test_neg_record_list),))) + list(np.ones(shape=(len(ECG_test_pos_record_list),)))
    ECG_test_name_record_label_list = list(zip(ECG_test_name_list, ECG_test_record_list, ECG_test_label_list))
    np.random.seed(3456)
    np.random.shuffle(ECG_test_name_record_label_list)
    ECG_test_name_list, ECG_test_record_list, ECG_test_label_list = zip(*ECG_test_name_record_label_list)

    print('ECG_test:', len(ECG_test_name_list), len(ECG_test_record_list), len(ECG_test_label_list))

    # PCG信号
    PCG_test_record_list = PCG_test_neg_record_list + PCG_test_pos_record_list
    PCG_test_name_list = PCG_test_neg_record_name_list + PCG_test_pos_record_name_list
    PCG_test_label_list = list(np.zeros(shape=(len(PCG_test_neg_record_list),))) + list(np.ones(shape=(len(PCG_test_pos_record_list),)))
    PCG_test_name_record_label_list = list(zip(PCG_test_name_list, PCG_test_record_list, PCG_test_label_list))
    np.random.seed(3456)
    np.random.shuffle(PCG_test_name_record_label_list)
    PCG_test_name_list, PCG_test_record_list, PCG_test_label_list = zip(*PCG_test_name_record_label_list)

    print('PCG_test:', len(PCG_test_name_list), len(PCG_test_record_list), len(PCG_test_label_list))

    # record分段 ECG信号
    ECG_test_patch_list = []
    ECG_test_patch_label_list = []
    ECG_test_patch_name_list = []
    ECG_test_num_list = []
    for i in range(len(ECG_test_record_list)):
        ECG_test_record = ECG_test_record_list[i]
        Frame_APEX = enframe(ECG_test_record[0][0, :], 2560, 1280)
        Frame_LLSB = enframe(ECG_test_record[1][0, :], 2560, 1280)
        Frame_LUSB = enframe(ECG_test_record[2][0, :], 2560, 1280)
        Frame_RUSB = enframe(ECG_test_record[3][0, :], 2560, 1280)
        Frame_APEX = Frame_APEX[..., np.newaxis]
        Frame_LLSB = Frame_LLSB[..., np.newaxis]
        Frame_LUSB = Frame_LUSB[..., np.newaxis]
        Frame_RUSB = Frame_RUSB[..., np.newaxis]  # (n,2560,1)

        Frame = np.concatenate((Frame_APEX, Frame_LLSB, Frame_LUSB, Frame_RUSB), axis=2)  # (n,2560,4)

        num = Frame.shape[0]
        for j in range(0, num, 1):
            ECG_test_patch = Frame[j, :, :]  # (2560,4)
            ECG_test_patch = ECG_test_patch[np.newaxis, ...]  # (1,2560,4)
            ECG_test_patch_list.append(ECG_test_patch)
            ECG_test_patch_label_list.append(ECG_test_label_list[i])
            ECG_test_patch_name_list.append(ECG_test_name_list[i])
        ECG_test_num_list.append(num)

    ECG_test_x = np.vstack(ECG_test_patch_list)  # (n,2560,4)
    ECG_test_y = np.vstack(ECG_test_patch_label_list)  # shape(n,1)
    ECG_test_record_label = np.vstack(ECG_test_label_list)

    # record分段 PCG信号
    PCG_test_patch_list = []
    PCG_test_patch_label_list = []
    PCG_test_patch_name_list = []
    PCG_test_num_list = []
    for i in range(len(PCG_test_record_list)):
        PCG_test_record = PCG_test_record_list[i]
        Frame_APEX = enframe(PCG_test_record[0][0, :], 2560, 1280)
        Frame_LLSB = enframe(PCG_test_record[1][0, :], 2560, 1280)
        Frame_LUSB = enframe(PCG_test_record[2][0, :], 2560, 1280)
        Frame_RUSB = enframe(PCG_test_record[3][0, :], 2560, 1280)
        Frame_APEX = Frame_APEX[..., np.newaxis]
        Frame_LLSB = Frame_LLSB[..., np.newaxis]
        Frame_LUSB = Frame_LUSB[..., np.newaxis]
        Frame_RUSB = Frame_RUSB[..., np.newaxis]  # (n,2560,1)

        Frame = np.concatenate((Frame_APEX, Frame_LLSB, Frame_LUSB, Frame_RUSB), axis=2)  # (n,2560,4)

        num = Frame.shape[0]
        for j in range(0, num, 1):
            PCG_test_patch = Frame[j, :, :]
            PCG_test_patch = PCG_test_patch[np.newaxis, ...]  # (1,2560,4)
            PCG_test_patch_list.append(PCG_test_patch)
            PCG_test_patch_label_list.append(PCG_test_label_list[i])
            PCG_test_patch_name_list.append(PCG_test_name_list[i])
        PCG_test_num_list.append(num)

    PCG_test_x = np.vstack(PCG_test_patch_list)  # shape(n,2560,4)
    PCG_test_y = np.vstack(PCG_test_patch_label_list)  # shape(n,1)
    PCG_test_record_label = np.vstack(PCG_test_label_list)

    ECG_test_x = ECG_test_x[..., np.newaxis]  # shape(n,2560,4,1)
    PCG_test_x = PCG_test_x[..., np.newaxis]  # shape(n,2560,4,1)
    ECG_PCG_test_x = np.concatenate((ECG_test_x, PCG_test_x), axis=3)  # shape(n,2560,4,2)

    if ((ECG_test_y.all() == PCG_test_y.all()) and (ECG_test_record_label.all() == PCG_test_record_label.all()) and \
            (ECG_test_num_list == PCG_test_num_list) and (ECG_test_patch_name_list == PCG_test_patch_name_list)):
        print('all equal')

    return ECG_PCG_test_x, ECG_test_y, ECG_test_record_label, ECG_test_num_list, ECG_test_patch_name_list, ECG_test_name_list
