import numpy as np
import math


def mean_std_computing(Sen_list, Spe_list, Acc_list, AUC_list, Ave_auc):
    # 心衰内部验证数据集
    # P_number_list = np.array([12,12,12,12,12])
    # N_number_list = np.array([88,88,88,88,88])

    # 心衰外部验证数据集
    # P_number_list = np.array([5, 5, 5, 5, 5])
    # N_number_list = np.array([115, 115, 115, 115, 115])

    # 公共数据集-全部样本
    P_number_list = np.array([57, 58, 57, 58, 58])
    N_number_list = np.array([23, 23, 24, 23, 24])

    Sen_list = np.array(Sen_list)
    Spe_list = np.array(Spe_list)
    Acc_list = np.array(Acc_list)
    AUC_list = np.array(AUC_list)
    Ave_auc = np.array(Ave_auc)

    Macc_list = (Sen_list + Spe_list) / 2
    print("Macc list", Macc_list)
    Ave_sen = sum(np.multiply(P_number_list, Sen_list)) / sum(P_number_list)
    Ave_spe = sum(np.multiply(N_number_list, Spe_list)) / sum(N_number_list)
    Ave_Acc = (sum(np.multiply(P_number_list, Sen_list)) + sum(np.multiply(N_number_list, Spe_list))) / (sum(P_number_list) + sum(N_number_list))
    Ave_macc = (Ave_sen + Ave_spe) / 2

    std_sen = math.sqrt(sum(np.square(Sen_list - Ave_sen)) / (len(Sen_list) - 1))
    std_spe = math.sqrt(sum(np.square(Spe_list - Ave_spe)) / (len(Spe_list) - 1))
    std_acc = math.sqrt(sum(np.square(Acc_list - Ave_Acc)) / (len(Acc_list) - 1))
    std_macc = math.sqrt(sum(np.square(Macc_list - Ave_macc)) / (len(Macc_list) - 1))
    std_auc = math.sqrt(sum(np.square(AUC_list - Ave_auc)) / (len(AUC_list) - 1))
    print("ave sen", Ave_sen * 100, "std sen", std_sen * 100)
    print("ave spe", Ave_spe * 100, "std spe", std_spe * 100)
    print("ave macc", Ave_macc * 100, "std macc", std_macc * 100)
    print("ave acc", Ave_Acc * 100, "std acc", std_acc * 100)
    print("ave auc", Ave_auc, "std auc", std_auc)
