# -*- coding: utf-8 -*-
import csv
import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, recall_score, specificity_score, balanced_accuracy_score

from source.data.load_data import load_data_cross_validate_train_test_four_locations_parallel_in_house_data
from source.models.LGC2_Net import LGC2_Net
from source.utils.mean_std_computing import mean_std_computing
from source.utils.path_tools import get_project_dir

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# path
project_dir = get_project_dir()
print("project dir:", project_dir)
data_dir = os.path.join(project_dir, "dataset/HFrEF_5_folds")
model_checkpoint_dir = os.path.join(project_dir, "model_checkpoints/LGC2_Net")
model_config = os.path.join(project_dir, 'source/models/LGC2_Net_config.json')
log_dir = os.path.join(project_dir, "logs/5_folds_cv")

k_folds = 5
Area_sum = 0
Sen_list = []
Spe_list = []
Acc_list = []
Area_list = []
resample_frequency = 2000
EF_threshold = 0.4
entire_dataset_recording_name_result_list = []
model_name_list = ['model_fold_0.hdf5',
                   'model_fold_1.hdf5',
                   'model_fold_2.hdf5',
                   'model_fold_3.hdf5',
                   'model_fold_4.hdf5']
for i in range(k_folds):
    ECG_PCG_train_x, ECG_train_y, ECG_PCG_test_x, ECG_test_y, ECG_test_record_label, ECG_test_num_list, ECG_test_patch_name_list, ECG_test_recording_name_list = \
        load_data_cross_validate_train_test_four_locations_parallel_in_house_data(data_dir, k_folds, i, resample_frequency, EF_threshold)
    print('train_x=', ECG_PCG_train_x.shape)
    print('train_y=', ECG_train_y.shape)
    print('test_x=', ECG_PCG_test_x.shape)
    print('test_y=', ECG_test_y.shape)

    tf.keras.backend.clear_session()
    result_save_dir = os.path.join(log_dir, f"fold_{i}")
    os.makedirs(result_save_dir, exist_ok=True)
    params = json.load(open(model_config, 'r'))
    model = LGC2_Net(**params)
    save_name = os.path.join(model_checkpoint_dir, model_name_list[i])
    model.load_weights(save_name)

    threshold_list = []
    FPR_list = []
    TP_list = []
    TN_list = []
    fold_pos_list = []
    fold_neg_list = []

    start_time = time.time()
    test_ECG_PCG_fusion_output, test_ECG_PCG_APEX_output, test_ECG_PCG_LLSB_output, test_ECG_PCG_LUSB_output, test_ECG_PCG_RUSB_output, test_ECG_PCG_channel_share_output = model.predict(
        ECG_PCG_test_x, batch_size=128)
    duration = time.time() - start_time
    time_per_seg = duration / ECG_PCG_test_x.shape[0]
    print("prediction time:", duration, "sec")
    print("time per seg:", time_per_seg)
    predict_probability = (test_ECG_PCG_fusion_output + test_ECG_PCG_APEX_output + test_ECG_PCG_LLSB_output + test_ECG_PCG_LUSB_output + test_ECG_PCG_RUSB_output + test_ECG_PCG_channel_share_output) / 6
    predict_patch_one_hot = np.argmax(predict_probability, axis=-1)
    ground_truth_list = (ECG_test_record_label[:, 0].astype(np.int8)).tolist()
    number_start_test = 0
    threshold_fix = 0.12
    pred_probability_test = []
    pred_list_test = []

    for k in range(0, len(ECG_test_num_list), 1):
        wav_length = ECG_test_num_list[k]
        normal = predict_probability[number_start_test:number_start_test + wav_length, 0].sum()
        abnormal = predict_probability[number_start_test:number_start_test + wav_length, 1].sum()
        summury = abnormal + normal
        if abnormal > summury * threshold_fix:
            case = 1
        if abnormal < summury * threshold_fix:
            case = 0
        if abnormal == summury * threshold_fix:
            case = 1
        pred_list_test.append(case)
        pred_probability_test.append(abnormal / wav_length)
        number_start_test = number_start_test + wav_length

    pred_array_test = np.hstack(pred_list_test).reshape(-1, 1)
    Y_test = ECG_test_record_label[:, 0].astype(np.int8)
    acc_test = accuracy_score(Y_test, pred_array_test)
    balance_acc_test = balanced_accuracy_score(Y_test, pred_array_test)
    Sen_test = recall_score(Y_test, pred_array_test)
    Spe_test = specificity_score(Y_test, pred_array_test)
    print(acc_test, balance_acc_test, Sen_test, Spe_test)
    Sen_list.append(Sen_test)
    Spe_list.append(Spe_test)
    Acc_list.append(acc_test)

    test_recording_name_result_list = list(zip(ECG_test_recording_name_list, ground_truth_list, pred_probability_test, pred_list_test))
    test_recording_name_result_list.sort()
    ECG_test_recording_name_list, ground_truth_list, pred_probability_test, pred_list_test = zip(*test_recording_name_result_list)
    entire_dataset_recording_name_result_list = entire_dataset_recording_name_result_list + test_recording_name_result_list
    # 保存每一折结果
    with open(os.path.join(result_save_dir, 'recoding_result.csv'), 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(['Recording_name', 'Ground_truth', 'Predict_result'])
        for j in range(len(ECG_test_recording_name_list)):
            data_row = [ECG_test_recording_name_list[j], ground_truth_list[j], pred_list_test[j]]
            csv_write.writerow(data_row)
        f.close()

    threshold = 1
    while True:
        pred_list_test = []
        number_start_test = 0
        for k in range(0, len(ECG_test_num_list), 1):
            wav_length = ECG_test_num_list[k]
            normal = 0
            abnormal = 0
            normal = predict_probability[number_start_test:number_start_test + wav_length, 0].sum()
            abnormal = predict_probability[number_start_test:number_start_test + wav_length, 1].sum()
            summury = abnormal + normal
            if abnormal > summury * threshold:
                case = 1
            if abnormal < summury * threshold:
                case = 0
            if abnormal == summury * threshold:
                case = 1
            pred_list_test.append(case)
            number_start_test = number_start_test + wav_length
        pred_array_test = np.hstack(pred_list_test).reshape(-1, 1)
        # 案例水平
        Y_test = ECG_test_record_label[:, 0].astype(np.int8)
        f1_test = f1_score(Y_test, pred_array_test, average="macro")
        acc_test = accuracy_score(Y_test, pred_array_test)
        balance_acc_test = balanced_accuracy_score(Y_test, pred_array_test)
        Sen_test = recall_score(Y_test, pred_array_test)
        Spe_test = specificity_score(Y_test, pred_array_test)
        print('Accuracy score_test=%.4f,Balanced accuracy=%.4f, score,Sensitivity score=%.4f,Specificity score=%.4f' % (acc_test, balance_acc_test, Sen_test, Spe_test))
        threshold_list.append(threshold)
        fold_pos = pred_array_test[Y_test == 1]
        fold_pos_list.append(fold_pos.shape[0])
        fold_neg = pred_array_test[Y_test == 0]
        fold_neg_list.append(fold_neg.shape[0])
        TP_pred = fold_pos[fold_pos == 1]
        TP_list.append(TP_pred.shape[0])
        TN_pred = fold_neg[fold_neg == 0]
        TN_list.append(TN_pred.shape[0])

        if threshold == 0:
            break

        if threshold > 0.01:
            threshold = threshold - 0.001
        elif (threshold <= 0.01) and (threshold > 0.001):
            threshold = threshold - 0.0001
        elif (threshold <= 0.001) and (threshold > 0.0001):
            threshold = threshold - 0.00001
        elif (threshold <= 0.0001) and (threshold > 0.00001):
            threshold = threshold - 0.000001
        elif (threshold <= 0.00001) and (threshold > 1e-9):
            threshold = threshold / 2
        elif threshold < 1e-9:
            threshold = 0

    threshold_array = np.hstack(threshold_list).reshape(-1, 1)
    fold_pos_array = np.hstack(fold_pos_list).reshape(-1, 1)
    fold_neg_array = np.hstack(fold_neg_list).reshape(-1, 1)
    TP_array = np.hstack(TP_list).reshape(-1, 1)
    TN_array = np.hstack(TN_list).reshape(-1, 1)
    acc_array = (TP_array + TN_array) / (fold_pos_array + fold_neg_array)
    Sen_array = TP_array / fold_pos_array
    Spe_array = TN_array / fold_neg_array
    Macc_array = (Sen_array + Spe_array) / 2

    result_index = np.concatenate((threshold_array, acc_array, Macc_array, Sen_array, Spe_array), axis=1)
    result_index_save_name = os.path.join(result_save_dir, 'threshold_Acc_Macc_Sen_Spe.csv')
    np.savetxt(result_index_save_name, result_index, delimiter=',')

    FPR_array = 1 - Spe_array
    FPR_Sensitivity_array = np.concatenate((FPR_array, Sen_array), axis=1)

    if i == 0:
        fold_pos_array_all = fold_pos_array
        fold_neg_array_all = fold_neg_array
        TP_array_all = TP_array
        TN_array_all = TN_array
    else:
        fold_pos_array_all = np.concatenate((fold_pos_array_all, fold_pos_array), axis=1)
        fold_neg_array_all = np.concatenate((fold_neg_array_all, fold_neg_array), axis=1)
        TP_array_all = np.concatenate((TP_array_all, TP_array), axis=1)
        TN_array_all = np.concatenate((TN_array_all, TN_array), axis=1)

    FPR_save_name = os.path.join(result_save_dir, 'FPR.csv')
    np.savetxt(FPR_save_name, FPR_array, delimiter=',')
    Sensitivity_array_save_name = os.path.join(result_save_dir, 'Sensitivity.csv')
    np.savetxt(Sensitivity_array_save_name, Sen_array, delimiter=',')
    Area = np.array([metrics.auc(FPR_array, Sen_array)])
    Area_save_name = os.path.join(result_save_dir, 'AUC.csv')
    np.savetxt(Area_save_name, Area, delimiter=',')
    Area_sum = Area_sum + Area
    Area_list.append(Area)

fold_pos_array_all = fold_pos_array_all.sum(axis=1).reshape(-1, 1)
fold_neg_array_all = fold_neg_array_all.sum(axis=1).reshape(-1, 1)
TP_array_all = TP_array_all.sum(axis=1).reshape(-1, 1)
TN_array_all = TN_array_all.sum(axis=1).reshape(-1, 1)
Acc_ave = (TN_array_all + TP_array_all) / (fold_neg_array_all + fold_pos_array_all)
Sen_ave = TP_array_all / fold_pos_array_all
Spe_ave = TN_array_all / fold_neg_array_all
MAcc_ave = (Sen_ave + Spe_ave) / 2
FPR_ave = 1 - Spe_ave
Ave_FPR_Sen = np.concatenate((FPR_ave, Sen_ave, Spe_ave, Acc_ave, MAcc_ave), axis=1)
Ave_FPR_Sen_save_name = os.path.join(log_dir, 'ECG_PCG_AVE_FPR_SEN_SPE_ACC_MACC_loss_weight_best_model.csv')
np.savetxt(Ave_FPR_Sen_save_name, Ave_FPR_Sen, delimiter=',')
print(Area_list)
Area_ave = np.array([metrics.auc(FPR_ave, Sen_ave)])
print(Area_ave)
Area_ave_save_name = os.path.join(log_dir, 'ECG_PCG_Ave_AUC_loss_weight_best_model.csv')
np.savetxt(Area_ave_save_name, Area_ave, delimiter=',')
# 保存整个数据集每个案例的预测结果
entire_dataset_recording_name_result_list.sort()
entire_dataset_recording_name_list, entire_dataset_ground_truth_list, entire_dataset_pred_probability_list, entire_dataset_pred_list = zip(*entire_dataset_recording_name_result_list)
with open(os.path.join(log_dir, 'recoding_result.csv'), 'w', newline='') as f:
    csv_write = csv.writer(f)
    csv_write.writerow(['Recording_name', 'Ground_truth', 'Predict_result'])
    for j in range(len(entire_dataset_recording_name_list)):
        data_row = [entire_dataset_recording_name_list[j], entire_dataset_ground_truth_list[j], entire_dataset_pred_list[j]]
        csv_write.writerow(data_row)
    f.close()

entire_dataset_ground_truth_array = np.array(entire_dataset_ground_truth_list)
entire_dataset_pred_probability_array = np.array(entire_dataset_pred_probability_list)
test_area_by_sklearn = metrics.roc_auc_score(entire_dataset_ground_truth_array, entire_dataset_pred_probability_array)
print('AUC_by_sklearn=', test_area_by_sklearn)
mean_std_computing(Sen_list, Spe_list, Acc_list, Area_list, Area_ave)
