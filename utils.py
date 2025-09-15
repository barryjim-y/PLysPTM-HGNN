#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn as nn
import torch.nn.functional as F
from logzero import logger


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        probs = torch.sigmoid(inputs)


        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')


        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)


        focal_weight = alpha_weight * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss






def separate_features(features):


    n_ipr = features.shape[1] - 400 - 1024

    features_dict = {
        'ipr': features[:, :n_ipr],
        'pssm': features[:, n_ipr:n_ipr + 400],
        'llm': features[:, n_ipr + 400:]
    }


    logger.info(f"  IPR: {features_dict['ipr'].shape}")
    logger.info(f"  PSSM: {features_dict['pssm'].shape}")
    logger.info(f"  llm: {features_dict['llm'].shape}")

    return features_dict


def load_processed_features(features_path):

    logger.info(f"loading {features_path} ")
    features = np.load(features_path)
    logger.info(f"feature: {features.shape}")
    return features


def prepare_imbalanced_data(labels, train_idx, approach='weighted', pos_ratio=0.5):


    flat_labels = labels.flatten()
    train_labels = flat_labels[train_idx]


    pos_count = train_labels.sum()
    neg_count = len(train_labels) - pos_count
    imbalance_ratio = neg_count / pos_count if pos_count > 0 else float('inf')

    logger.info(f"Training set label statistics - Total samples: {len(train_idx)}, Positives: {pos_count} ({pos_count / len(train_idx):.2%}), "
                f"Negatives: {neg_count} ({neg_count / len(train_idx):.2%}), Imbalance ratio: {imbalance_ratio:.2f}")

    if approach == 'weighted':

        pos_weight = torch.tensor(imbalance_ratio, dtype=torch.float32)
        loss_weight = pos_weight
        logger.info(f"Using weighted loss - Positive weight: {pos_weight.item():.4f}")
        return train_idx, loss_weight

    elif approach == 'undersample':

        pos_indices = [i for i, idx in enumerate(train_idx) if flat_labels[idx] == 1]
        neg_indices = [i for i, idx in enumerate(train_idx) if flat_labels[idx] == 0]

        target_neg_count = int(pos_count / pos_ratio) - pos_count
        target_neg_count = min(target_neg_count, len(neg_indices))


        sampled_neg_indices = random.sample(neg_indices, target_neg_count)
        balanced_indices = pos_indices + sampled_neg_indices
        random.shuffle(balanced_indices)


        balanced_train_idx = [train_idx[i] for i in balanced_indices]

        logger.info(f"After undersampling - Total samples: {len(balanced_train_idx)}, "
                    f"Positive ratio: {pos_count / len(balanced_train_idx):.2%}")
        return balanced_train_idx, torch.tensor(1.0)  # No additional weight needed

    elif approach == 'oversample':

        sample_weights = torch.ones(len(train_idx), dtype=torch.float32)
        for i, idx in enumerate(train_idx):
            if flat_labels[idx] == 1:  # Positive
                sample_weights[i] = imbalance_ratio

        logger.info(f"Using oversampling - Positive sampling weight: {imbalance_ratio:.4f}")
        return train_idx, sample_weights

    elif approach == 'mixed':

        pos_indices = [i for i, idx in enumerate(train_idx) if flat_labels[idx] == 1]
        neg_indices = [i for i, idx in enumerate(train_idx) if flat_labels[idx] == 0]


        target_ratio = 5.0
        target_neg_count = int(pos_count * target_ratio)
        target_neg_count = min(target_neg_count, len(neg_indices))

        sampled_neg_indices = random.sample(neg_indices, target_neg_count)
        balanced_indices = pos_indices + sampled_neg_indices
        random.shuffle(balanced_indices)


        balanced_train_idx = [train_idx[i] for i in balanced_indices]


        new_imbalance_ratio = target_ratio
        pos_weight = torch.tensor(new_imbalance_ratio, dtype=torch.float32)

        logger.info(f"Mixed strategy - After undersampling total samples: {len(balanced_train_idx)}, "
                    f"Positive ratio: {pos_count / len(balanced_train_idx):.2%}, "
                    f"Positive loss weight: {pos_weight.item():.4f}")
        return balanced_train_idx, pos_weight

    else:
        logger.warning(f"Unknown imbalance handling method: {approach}, using raw data")
        return train_idx, torch.tensor(1.0)


def get_norm_net_mat(net_mat):


    degree_0 = np.asarray(net_mat.sum(0)).squeeze()
    degree_0_safe = np.maximum(degree_0, 1e-12)  # Prevent division by zero
    mat_d_0 = ssp.diags(degree_0_safe ** -0.5, format='csr')


    degree_1 = np.asarray(net_mat.sum(1)).squeeze()
    degree_1_safe = np.maximum(degree_1, 1e-12)  # Prevent division by zero
    mat_d_1 = ssp.diags(degree_1_safe ** -0.5, format='csr')

    # D^(-1/2) A D^(-1/2) normalization
    return mat_d_0 @ net_mat @ mat_d_1


def load_go_features(feature_path):

    logger.info(f"Loading GO features: {feature_path}")
    if feature_path.endswith('.npz'):
        return ssp.load_npz(feature_path)
    else:
        features = np.load(feature_path)
        logger.info(f"GO feature matrix: {features.shape}")
        return features


def load_sparse_features(feature_path):


    if feature_path.endswith('.npz'):
        logger.info(f"Loading npz sparse features: {feature_path}")
        sparse_data = np.load(feature_path, allow_pickle=True)

        required_keys = ['indices', 'indptr', 'shape', 'data']
        for key in required_keys:
            if key not in sparse_data:
                raise ValueError(f"Sparse feature file missing required key '{key}'")


        if 'format' in sparse_data:
            matrix_format = sparse_data['format']
            if isinstance(matrix_format, np.ndarray):
                matrix_format = matrix_format.item()
            if isinstance(matrix_format, bytes):
                matrix_format = matrix_format.decode('utf-8')
        else:
            matrix_format = 'csr'


        if matrix_format == 'csr':
            return ssp.csr_matrix(
                (sparse_data['data'], sparse_data['indices'], sparse_data['indptr']),
                shape=tuple(sparse_data['shape'])
            )
        elif matrix_format == 'csc':
            return ssp.csc_matrix(
                (sparse_data['data'], sparse_data['indices'], sparse_data['indptr']),
                shape=tuple(sparse_data['shape'])
            )
        else:
            raise ValueError(f"Unsupported sparse matrix format: {matrix_format}")
    else:
        logger.info(f"Loading npy sparse features: {feature_path}")
        sparse_features = np.load(feature_path, allow_pickle=True).item()
        # Rebuild CSR matrix
        return ssp.csr_matrix(
            (sparse_features['data'], sparse_features['indices'], sparse_features['indptr']),
            shape=sparse_features['shape']
        )


def preprocess_features(features, train_indices=None):


    if ssp.issparse(features):
        dense_features = features.todense()
    else:
        dense_features = features


    if train_indices is not None:
        logger.info("Using training set only to compute feature statistics")
        train_features = dense_features[train_indices]
        mean = np.mean(train_features, axis=0)
        std = np.std(train_features, axis=0)
    else:

        logger.info("Using all data to compute feature statistics")
        mean = np.mean(dense_features, axis=0)
        std = np.std(dense_features, axis=0)


    std_safe = np.maximum(std, 1e-8)


    norm_features = (dense_features - mean) / std_safe


    norm_features = np.clip(norm_features, -5, 5)


    if np.isnan(norm_features).any() or np.isinf(norm_features).any():
        logger.warning("NaN or Inf found in normalized features, replacing with zero")
        norm_features = np.nan_to_num(norm_features, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Feature normalization - Mean: {np.mean(norm_features):.4f}, Std: {np.std(norm_features):.4f}, "
                f"Min: {np.min(norm_features):.4f}, Max: {np.max(norm_features):.4f}")

    return norm_features


def preprocess_all_features(go_features, pssm_features, llm_features, train_indices=None):


    processed_go = preprocess_features(go_features, train_indices)


    if train_indices is not None:
        # Compute statistics based only on training set
        train_pssm = pssm_features[train_indices]
        pssm_mean = np.mean(train_pssm, axis=0)
        pssm_std = np.std(train_pssm, axis=0)
    else:
        pssm_mean = np.mean(pssm_features, axis=0)
        pssm_std = np.std(pssm_features, axis=0)


    pssm_std_safe = np.maximum(pssm_std, 1e-8)
    processed_pssm = (pssm_features - pssm_mean) / pssm_std_safe


    processed_pssm = np.clip(processed_pssm, -5, 5)
    processed_pssm = np.nan_to_num(processed_pssm, nan=0.0, posinf=0.0, neginf=0.0)


    if train_indices is not None:

        train_llm = llm_features[train_indices]
        llm_mean = np.mean(train_llm, axis=0)
        llm_std = np.std(train_llm, axis=0)
    else:
        llm_mean = np.mean(llm_features, axis=0)
        llm_std = np.std(llm_features, axis=0)


    llm_std_safe = np.maximum(llm_std, 1e-8)
    processed_llm = (llm_features - llm_mean) / llm_std_safe



    processed_llm = np.clip(processed_llm, -5, 5)
    processed_llm = np.nan_to_num(processed_llm, nan=0.0, posinf=0.0, neginf=0.0)


    features_dict = {
        'ipr': processed_go,
        'pssm': processed_pssm,
        'llm': processed_llm
    }

    logger.info(f"Feature preprocessing completed:")
    logger.info(f"  GO features: {processed_go.shape}")
    logger.info(f"  PSSM features: {processed_pssm.shape}")
    logger.info(f"  LLM features: {processed_llm.shape}")

    return features_dict


def load_labels(labels_path):

    if labels_path.endswith('.npz'):
        logger.info(f"Loading labels from npz file: {labels_path}")

        loaded_data = np.load(labels_path, allow_pickle=True)
        if 'labels' in loaded_data:
            return loaded_data['labels']
        elif 'arr_0' in loaded_data:
            return loaded_data['arr_0']
        else:
            # Try to fetch the first array
            for key in loaded_data:
                return loaded_data[key]
            raise ValueError("Could not find label data in npz file")
    else:
        logger.info(f"Loading labels from npy file: {labels_path}")
        return np.load(labels_path, allow_pickle=True)