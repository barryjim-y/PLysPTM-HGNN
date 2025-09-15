#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp as amp
from tqdm import tqdm, trange
from datetime import datetime
import matplotlib.pyplot as plt
from logzero import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from utils import preprocess_all_features, set_seed, prepare_imbalanced_data, FocalLoss
from eval import multi_evaluate_multilabel,macro_roc_curve, macro_pr_curve,plot_ptm_roc_pr_curves
from model import MultiFeatureGCN

def load_ptm_names(ptm_file):
    with open(ptm_file, 'r', encoding='utf-8') as f:
        ptm_names = [line.strip() for line in f.readlines()]
    return ptm_names



def train_protein_model_multi_feature(
        model, pyg_graph, features_dict, labels, train_proteins, val_proteins, device,
        epochs=100, lr=0.001, weight_decay=1e-6,
        imbalance_approach='weighted', focal_loss=False, use_amp=True):


    model.to(device)

    ptm_names = load_ptm_names('lysine_site.txt')


    pyg_graph = pyg_graph.to(device)
    features_tensor_dict = {
        'ipr': torch.FloatTensor(features_dict['ipr']).to(device),
        'pssm': torch.FloatTensor(features_dict['pssm']).to(device),
        'llm': torch.FloatTensor(features_dict['llm']).to(device)
    }


    labels_tensor = torch.FloatTensor(labels)


    train_labels = labels_tensor[train_proteins]
    val_labels = labels_tensor[val_proteins]


    train_pos = train_labels.sum().item()
    train_total = train_labels.numel()
    train_neg = train_total - train_pos

    logger.info(f"Training set: Positives {train_pos}, Negatives {train_neg}, Positive ratio {train_pos / train_total:.4f}")


    pos_weight = torch.tensor(train_neg / train_pos if train_pos > 0 else 1.0)


    if focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        pos_weight = pos_weight.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    ipr_params = []
    pssm_params = []
    llm_params = []
    interaction_params = []
    other_params = []


    for name, param in model.named_parameters():
        if 'interaction' in name:

            interaction_params.append(param)
        elif 'ipr_transform' in name or 'gcn_layers_ipr' in name or 'layer_norms_ipr' in name:

            ipr_params.append(param)
        elif 'pssm_transform' in name or 'gcn_layers_pssm' in name or 'layer_norms_pssm' in name:

            pssm_params.append(param)
        elif 'llm_transform' in name or 'gcn_layers_llm' in name or 'layer_norms_llm' in name:

            llm_params.append(param)
        else:

            other_params.append(param)

    optimizer = optim.AdamW([
        {'params': ipr_params, 'lr': lr * 0.1, 'weight_decay': weight_decay * 2.0},  # Further reduce IPR learning rate, increase regularization
        {'params': pssm_params, 'lr': lr * 2.5, 'weight_decay': weight_decay * 0.2},  # Increase PSSM learning rate, reduce regularization
        {'params': llm_params, 'lr': lr * 0.6, 'weight_decay': weight_decay * 1.0},  # Adjust LLM parameters moderately
        {'params': interaction_params, 'lr': lr * 1.4, 'weight_decay': weight_decay * 0.35},  # Increase interaction layer learning rate
        {'params': other_params, 'lr': lr * 1.2, 'weight_decay': weight_decay * 0.5}  # Slightly increase other layers learning rate
    ], lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)


    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=8, T_mult=2, eta_min=lr / 2000  # Lower final learning rate
    )


    scaler = amp.GradScaler() if use_amp else None


    best_val_f1 = 0.0
    best_val_g_mean = 0.0
    early_stop_count = 0
    early_stop_patience = 20
    best_model_state = None

    history = {
        'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_g_mean': [],
        'val_precision': [], 'val_recall': [], 'val_roc_auc': [], 'val_pr_auc': [],
        'grad_norm': []
    }


    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()

        if use_amp:
            with amp.autocast():

                outputs = model(pyg_graph, features_tensor_dict)


                train_outputs = outputs[train_proteins]


                loss = criterion(train_outputs, train_labels.to(device))


            scaler.scale(loss).backward()


            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()


            if grad_norm < 1e-5:
                logger.warning(f"Gradient too small: {grad_norm:.8f}, scaled up 10x")
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(10.0)
                grad_norm *= 10


            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(pyg_graph, features_tensor_dict)

            train_outputs = outputs[train_proteins]
            loss = criterion(train_outputs, train_labels.to(device))

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()


            if grad_norm < 1e-5:
                logger.warning(f"Gradient too small: {grad_norm:.8f}, scaled up 10x")
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(10.0)
                grad_norm *= 10

            optimizer.step()


        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']


        history['train_loss'].append(loss.item())
        history['grad_norm'].append(grad_norm)


        model.eval()
        with torch.no_grad():
            outputs = model(pyg_graph, features_tensor_dict)
            val_outputs = outputs[val_proteins]
            val_loss = criterion(val_outputs, val_labels.to(device))


            val_probs = torch.sigmoid(val_outputs).cpu().numpy()
            val_true = val_labels.numpy()

            ptm_names = load_ptm_names('lysine_site.txt')
            plot_ptm_roc_pr_curves(val_probs, val_true, ptm_names, save_prefix="cv_ptm")




            metrics = multi_evaluate_multilabel(val_probs, val_true)


            history['val_loss'].append(val_loss.item())
            history['val_precision'].append(metrics['precision_micro'])
            history['val_recall'].append(metrics['recall_micro'])
            history['val_f1'].append(metrics['f1_micro'])
            history['val_g_mean'].append(0.0)
            history['val_roc_auc'].append(metrics['roc_auc'])
            history['val_pr_auc'].append(metrics['pr_auc'])


            log_message = f"Epoch {epoch + 1}/{epochs} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, "
            log_message += f"F1: {metrics['f1_micro']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}, "
            log_message += f"Grad norm: {grad_norm:.6f}, LR: {current_lr:.6f}"

            logger.info(log_message)

            if epoch % 5 == 0:

                logger.info(f"Using raw feature fusion, no feature importance weights")


            current_score = (metrics['f1_micro'] + metrics['roc_auc']) / 2
            best_score = (best_val_f1 + best_val_g_mean) / 2

            if current_score > best_score:
                best_val_f1 = metrics['f1_micro']
                best_val_g_mean = metrics['roc_auc']
                early_stop_count = 0

                best_model_state = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'metrics': metrics
                }

                logger.info(
                    f"Epoch {epoch + 1} - New best model! F1-micro: {metrics['f1_micro']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")

            else:
                early_stop_count += 1


            if early_stop_count >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}, no improvement for {early_stop_patience} rounds")
                break


    if best_model_state:
        model.load_state_dict(best_model_state['model'])
        best_metrics = best_model_state['metrics']
    else:
        logger.warning("No best model found, using final model")
        model.eval()
        with torch.no_grad():
            outputs = model(pyg_graph, features_tensor_dict)
            val_outputs = outputs[val_proteins]
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()
            val_true = val_labels.numpy()
            best_metrics = multi_evaluate_multilabel(val_probs, val_true)


    plt.figure(figsize=(16, 12))


    plt.subplot(3, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')


    plt.subplot(3, 2, 2)
    plt.plot(history['val_f1'], label='F1 Score')
    plt.plot(history['val_roc_auc'], label='ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('F1 and ROC AUC')


    plt.subplot(3, 2, 3)
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Precision and Recall')


    plt.subplot(3, 2, 4)
    plt.plot(history['grad_norm'], label='Gradient L2 Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    plt.title('Gradient Norm')
    plt.yscale('log')


    if 'feature_weights_ipr' in history:

        if len(history['feature_weights_ipr']) > 1:

            weight_record_interval = len(history['train_loss']) // (len(history['feature_weights_ipr']) - 1)
            x_ticks = range(0, len(history['train_loss']), weight_record_interval)[:len(history['feature_weights_ipr'])]
        else:
            x_ticks = [0]


        plt.subplot(3, 2, 5)
        plt.plot(x_ticks, history['feature_weights_ipr'], label='IPR')
        plt.plot(x_ticks, history['feature_weights_pssm'], label='PSSM')
        plt.plot(x_ticks, history['feature_weights_llm'], label='LLM')
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        plt.legend()
        plt.title('Feature Importance Weights')

    plt.tight_layout()
    plt.savefig('training_curves_multi_feature.png')


    return model, best_metrics, history


def train_with_cross_validation_multi_feature(
        model_class, pyg_graph, go_features, pssm_features, llm_features, labels, device,
        n_folds=5, epochs=100, lr=0.001, batch_size=64,
        hidden_dim=256, n_layers=2, dropout=0.3, weight_decay=1e-6,
        imbalance_approach='weighted', focal_loss=False, use_amp=True):


    set_seed(42)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./imbalanced_multi_feature_outputs_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    n_proteins = labels.shape[0]
    n_modifications = labels.shape[1]
    logger.info(f"Data shape - Number of proteins: {n_proteins}, Number of modification types: {n_modifications}")

    protein_indices = np.arange(n_proteins)
    protein_label_means = np.mean(labels, axis=1)
    stratify_values = (protein_label_means > np.mean(protein_label_means)).astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    models = []
    all_metrics = []
    all_val_probs = []
    all_val_trues = []
    per_fold_roc = []
    per_fold_pr = []

    for fold_idx, (train_protein_idx, val_protein_idx) in enumerate(skf.split(protein_indices, stratify_values)):
        logger.info(f"Starting fold {fold_idx + 1}/{n_folds}")
        train_proteins = torch.LongTensor(train_protein_idx)
        val_proteins = torch.LongTensor(val_protein_idx)

        features_dict = preprocess_all_features(go_features, pssm_features, llm_features, train_protein_idx)
        input_dims = {
            'ipr': features_dict['ipr'].shape[1],
            'pssm': features_dict['pssm'].shape[1],
            'llm': features_dict['llm'].shape[1]
        }
        output_dim = n_modifications

        model = model_class(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            dropout=dropout
        )

        trained_model, fold_metrics, history = train_protein_model_multi_feature(
            model, pyg_graph, features_dict, labels,
            train_proteins, val_proteins, device,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            imbalance_approach=imbalance_approach,
            focal_loss=focal_loss, use_amp=use_amp
        )

        with torch.no_grad():
            features_tensor_dict = {
                'ipr': torch.FloatTensor(features_dict['ipr']).to(device),
                'pssm': torch.FloatTensor(features_dict['pssm']).to(device),
                'llm': torch.FloatTensor(features_dict['llm']).to(device)
            }
            outputs = trained_model(pyg_graph, features_tensor_dict)
            val_outputs = outputs[val_proteins]
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()
            val_true = labels[val_proteins]


            ptm_names = load_ptm_names('lysine_site.txt')
            plot_ptm_roc_pr_curves(val_probs, val_true, ptm_names, save_prefix=f"cv_fold_{fold_idx + 1}_ptm")


            lysine_types_per_protein = val_true.sum(axis=1)
            multi_lysine_mask = lysine_types_per_protein >= 5
            few_lysine_mask = lysine_types_per_protein < 5


            fold_dump_dir = os.path.join(output_dir, "per_fold_scores")
            os.makedirs(fold_dump_dir, exist_ok=True)


            np.savez_compressed(
                os.path.join(fold_dump_dir, f"fold_{fold_idx + 1}.npz"),
                val_probs=val_probs,  # [N_val, n_labels]
                val_true=val_true,  # [N_val, n_labels]
                lysine_types_per_protein=lysine_types_per_protein,  # [N_val]
                val_indices=val_protein_idx

            )


            if np.sum(multi_lysine_mask) > 0:
                metrics_multi = multi_evaluate_multilabel(val_probs[multi_lysine_mask], val_true[multi_lysine_mask])
                fold_metrics['roc_auc_multi_lysine'] = metrics_multi['roc_auc']
                fold_metrics['pr_auc_multi_lysine'] = metrics_multi['pr_auc']
            else:
                fold_metrics['roc_auc_multi_lysine'] = float('nan')
                fold_metrics['pr_auc_multi_lysine'] = float('nan')

            if np.sum(few_lysine_mask) > 0:
                metrics_few = multi_evaluate_multilabel(val_probs[few_lysine_mask], val_true[few_lysine_mask])
                fold_metrics['roc_auc_few_lysine'] = metrics_few['roc_auc']
                fold_metrics['pr_auc_few_lysine'] = metrics_few['pr_auc']
            else:
                fold_metrics['roc_auc_few_lysine'] = float('nan')
                fold_metrics['pr_auc_few_lysine'] = float('nan')
            logger.info(f"Multi-lysine proteins (>=5): ROC-AUC={fold_metrics['roc_auc_multi_lysine']:.4f}, "
                        f"PR-AUC={fold_metrics['pr_auc_multi_lysine']:.4f}")
            logger.info(f"Few-lysine proteins (<5): ROC-AUC={fold_metrics['roc_auc_few_lysine']:.4f}, "
                        f"PR-AUC={fold_metrics['pr_auc_few_lysine']:.4f}")
            all_val_probs.append(val_probs)
            all_val_trues.append(val_true)

        models.append(trained_model)
        all_metrics.append(fold_metrics)
        per_fold_roc.append(fold_metrics.get('roc_auc', float('nan')))
        per_fold_pr.append(fold_metrics.get('pr_auc', float('nan')))

        logger.info(f"Fold {fold_idx + 1} completed - F1-micro: {fold_metrics['f1_micro']:.4f}, "
                    f"ROC AUC: {fold_metrics['roc_auc']:.4f}")


    metric_keys = ['precision_micro', 'recall_micro', 'f1_micro',
                   'precision_macro', 'recall_macro', 'f1_macro',
                   'roc_auc', 'pr_auc', 'accuracy', 'hamming_loss',
                   'aiming', 'coverage', 'set_accuracy', 'absolute_true', 'absolute_false',
                   'roc_auc_multi_lysine', 'pr_auc_multi_lysine', 'roc_auc_few_lysine', 'pr_auc_few_lysine']

    avg_metrics = {}
    std_metrics = {}

    for key in metric_keys:
        values = [m.get(key, float('nan')) for m in all_metrics]
        avg_metrics[key] = np.nanmean(values)
        std_metrics[key] = np.nanstd(values)

    logger.info("\n" + "=" * 20 + " Cross-validation Results " + "=" * 20)
    for key in metric_keys:
        logger.info(f"{key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")


    plot_cv_roc_pr_curves(
        all_val_probs, all_val_trues,
        per_fold_roc, per_fold_pr,
        save_prefix=f"{output_dir}/cv"
    )


    all_dump_dir = os.path.join(output_dir, "per_fold_scores")
    os.makedirs(all_dump_dir, exist_ok=True)


    all_probs_concat = np.concatenate(all_val_probs, axis=0)
    all_true_concat = np.concatenate(all_val_trues, axis=0)

    np.savez_compressed(
        os.path.join(all_dump_dir, "all_folds_concat.npz"),
        probs=all_probs_concat,  # [sum_N_val, n_labels]
        true=all_true_concat  # [sum_N_val, n_labels]
    )

    return models, avg_metrics, std_metrics


def macro_roc_curve(y_true: np.ndarray, y_score: np.ndarray, n_points: int = 400):
    fpr_grid = np.linspace(0.0, 1.0, n_points)
    tprs = []
    for k in range(y_true.shape[1]):
        yk = y_true[:, k]
        pk = y_score[:, k]
        if yk.min() == yk.max():
            continue
        fpr_k, tpr_k, _ = roc_curve(yk, pk)
        tprs.append(np.interp(fpr_grid, fpr_k, tpr_k))
    if not tprs:
        return fpr_grid, np.zeros_like(fpr_grid)
    return fpr_grid, np.mean(np.vstack(tprs), axis=0)

def macro_pr_curve(y_true: np.ndarray, y_score: np.ndarray, n_points: int = 400):
    recall_grid = np.linspace(0.0, 1.0, n_points)
    precs = []
    for k in range(y_true.shape[1]):
        yk = y_true[:, k]
        pk = y_score[:, k]
        if yk.min() == yk.max():
            continue
        precision_k, recall_k, _ = precision_recall_curve(yk, pk)

        recall_unique, idx = np.unique(recall_k, return_index=True)
        precision_unique = precision_k[idx]
        precs.append(np.interp(recall_grid, recall_unique, precision_unique))
    if not precs:
        return recall_grid, np.zeros_like(recall_grid)
    return recall_grid, np.mean(np.vstack(precs), axis=0)


def plot_cv_roc_pr_curves(all_val_probs, all_val_trues, per_fold_roc, per_fold_pr, save_prefix="cv_curves"):

    plt.figure(figsize=(8, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_val_probs)))


    for i, (probs, trues) in enumerate(zip(all_val_probs, all_val_trues)):
        fpr_grid, mean_tpr = macro_roc_curve(trues, probs, n_points=400)
        plt.plot(
            fpr_grid, mean_tpr, color=colors[i], lw=2,
            label=f'Fold {i+1} (AUROC={per_fold_roc[i]:.4f})'
        )
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0])
    plt.xticks(np.arange(0, 1.01, 0.2), fontsize=18, fontname="Times New Roman")
    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=18, fontname="Times New Roman")
    plt.xlabel("False Positive Rate", fontsize=20, fontname="Times New Roman")
    plt.ylabel("True Positive Rate", fontsize=20, fontname="Times New Roman")
    plt.title("Cross-Validation ROC Curves", fontsize=22, fontname="Times New Roman")
    plt.legend(prop={'size': 12, 'family': "Times New Roman"})
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_roc.pdf", bbox_inches="tight")
    plt.close()


    plt.figure(figsize=(8, 7))
    for i, (probs, trues) in enumerate(zip(all_val_probs, all_val_trues)):
        recall_grid, mean_precision = macro_pr_curve(trues, probs, n_points=400)
        plt.plot(
            recall_grid, mean_precision, color=colors[i], lw=2,
            label=f'Fold {i+1} (AUPR={per_fold_pr[i]:.4f})'
        )
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0])
    plt.xticks(np.arange(0, 1.01, 0.2), fontsize=18, fontname="Times New Roman")
    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=18, fontname="Times New Roman")
    plt.xlabel("Recall", fontsize=20, fontname="Times New Roman")
    plt.ylabel("Precision", fontsize=20, fontname="Times New Roman")
    plt.title("Cross-Validation PR Curves", fontsize=22, fontname="Times New Roman")
    plt.legend(prop={'size': 12, 'family': "Times New Roman"})
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_pr.pdf", bbox_inches="tight")
    plt.close()





