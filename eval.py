#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from logzero import logger
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, accuracy_score,roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
from matplotlib import rcParams

def multi_evaluate_multilabel(y_pred, y_test):

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.detach().cpu().numpy()


    y_pred_binary = (y_pred >= 0.5).astype(int)


    precision_micro = precision_score(y_test, y_pred_binary, average='micro', zero_division=0)
    recall_micro = recall_score(y_test, y_pred_binary, average='micro', zero_division=0)
    f1_micro = f1_score(y_test, y_pred_binary, average='micro', zero_division=0)

    precision_macro = precision_score(y_test, y_pred_binary, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred_binary, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred_binary, average='macro', zero_division=0)


    n_labels = y_test.shape[1]
    roc_auc_list = []
    pr_auc_list = []

    for i in range(n_labels):
        if len(np.unique(y_test[:, i])) > 1:  # Ensure both positive and negative examples exist
            try:
                roc_auc_list.append(roc_auc_score(y_test[:, i], y_pred[:, i]))
                pr_auc_list.append(average_precision_score(y_test[:, i], y_pred[:, i]))
            except:
                continue

    roc_auc = np.mean(roc_auc_list) if roc_auc_list else 0.0
    pr_auc = np.mean(pr_auc_list) if pr_auc_list else 0.0


    accuracy = np.mean(np.all(y_pred_binary == y_test, axis=1))


    hamming_loss = np.mean(y_pred_binary != y_test)


    N = len(y_test)
    M = y_test.shape[1]  # Number of labels

    count_Aiming = 0
    count_Coverage = 0
    count_Accuracy = 0
    Aiming = 0
    Coverage = 0
    Accuracy = 0
    Absolute_True = 0
    Absolute_False = 0

    for i in range(N):
        union_set_len = np.sum(np.maximum(y_pred_binary[i], y_test[i]))
        inter_set_len = np.sum(np.minimum(y_pred_binary[i], y_test[i]))
        y_pred_len = np.sum(y_pred_binary[i])
        y_test_len = np.sum(y_test[i])

        if y_pred_len > 0:
            Aiming += inter_set_len / y_pred_len
            count_Aiming = count_Aiming + 1

        if y_test_len > 0:
            Coverage += inter_set_len / y_test_len
            count_Coverage = count_Coverage + 1

        if union_set_len > 0:
            Accuracy += inter_set_len / union_set_len
            count_Accuracy = count_Accuracy + 1

        Absolute_True += int(np.array_equal(y_pred_binary[i], y_test[i]))
        Absolute_False += (union_set_len - inter_set_len) / M

    Aiming = Aiming / count_Aiming if count_Aiming > 0 else 0
    Coverage = Coverage / count_Coverage if count_Coverage > 0 else 0
    Accuracy = Accuracy / count_Accuracy if count_Accuracy > 0 else 0
    Absolute_True = Absolute_True / N
    Absolute_False = Absolute_False / N



    logger.info(f"  ROC AUC: {roc_auc:.4f}")
    logger.info(f"  PR AUC: {pr_auc:.4f}")


    return {
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": accuracy,
        "hamming_loss": hamming_loss,
        "aiming": Aiming,
        "coverage": Coverage,
        "set_accuracy": Accuracy,  # Use set_accuracy as key name to avoid confusion with accuracy
        "absolute_true": Absolute_True,
        "absolute_false": Absolute_False
    }

def macro_roc_curve(y_true: np.ndarray, y_score: np.ndarray, n_points: int = 200):

    fpr_grid = np.linspace(0.0, 1.0, n_points)
    tpr_stack = []

    n_labels = y_true.shape[1]
    for k in range(n_labels):
        yk_true = y_true[:, k]
        yk_score = y_score[:, k]


        if yk_true.min() == yk_true.max():
            continue

        fpr_k, tpr_k, _ = roc_curve(yk_true, yk_score)

        tpr_interp = np.interp(fpr_grid, fpr_k, tpr_k)
        tpr_stack.append(tpr_interp)

    if len(tpr_stack) == 0:

        return fpr_grid, np.zeros_like(fpr_grid)

    mean_tpr = np.mean(np.vstack(tpr_stack), axis=0)
    return fpr_grid, mean_tpr


def macro_pr_curve(y_true: np.ndarray, y_score: np.ndarray, n_points: int = 200):

    recall_grid = np.linspace(0.0, 1.0, n_points)
    prec_stack = []

    n_labels = y_true.shape[1]
    for k in range(n_labels):
        yk_true = y_true[:, k]
        yk_score = y_score[:, k]

        if yk_true.min() == yk_true.max():
            continue

        precision_k, recall_k, _ = precision_recall_curve(yk_true, yk_score)

        recall_k_unique, idx = np.unique(recall_k, return_index=True)
        precision_k_unique = precision_k[idx]

        prec_interp = np.interp(recall_grid, recall_k_unique, precision_k_unique)
        prec_stack.append(prec_interp)

    if len(prec_stack) == 0:
        return recall_grid, np.zeros_like(recall_grid)

    mean_precision = np.mean(np.vstack(prec_stack), axis=0)
    return recall_grid, mean_precision


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score


def plot_ptm_roc_pr_curves(val_probs, val_true, ptm_names, save_prefix="ptm_curves"):


    rcParams['font.family'] = 'Times New Roman'
    rcParams['axes.unicode_minus'] = False


    plt.figure(figsize=(10, 8))
    for i, ptm_name in enumerate(ptm_names):
        fpr, tpr, _ = roc_curve(val_true[:, i], val_probs[:, i])
        auc_score = roc_auc_score(val_true[:, i], val_probs[:, i])
        plt.plot(fpr, tpr, label=f'{ptm_name}  (AUROC={auc_score:.4f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate ', fontsize=20, fontname="Times New Roman")
    plt.ylabel('True Positive Rate ', fontsize=20, fontname="Times New Roman")
    plt.title('Cross-Validation ROC Curves', fontsize=22, fontname="Times New Roman")
    plt.xticks(np.arange(0, 1.01, 0.2), fontsize=18, fontname="Times New Roman")
    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=18, fontname="Times New Roman")
    plt.legend(prop={'size': 12, 'family': "Times New Roman"})
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.savefig(f"{save_prefix}_roc.pdf", bbox_inches="tight")
    plt.close()


    plt.figure(figsize=(10, 8))
    for i, ptm_name in enumerate(ptm_names):
        precision, recall, _ = precision_recall_curve(val_true[:, i], val_probs[:, i])
        pr_auc = average_precision_score(val_true[:, i], val_probs[:, i])
        plt.plot(recall, precision, label=f'{ptm_name}  (AUPR={pr_auc:.4f})')

    plt.xlabel('Recall', fontsize=20, fontname="Times New Roman")
    plt.ylabel('Precision', fontsize=20, fontname="Times New Roman")
    plt.title('Cross-Validation PR Curves', fontsize=22, fontname="Times New Roman")
    plt.xticks(np.arange(0, 1.01, 0.2), fontsize=18, fontname="Times New Roman")
    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=18, fontname="Times New Roman")
    plt.legend(prop={'size': 12, 'family': "Times New Roman"})
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.savefig(f"{save_prefix}_pr.pdf", bbox_inches="tight")
    plt.close()
