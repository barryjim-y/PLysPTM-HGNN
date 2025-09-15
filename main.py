#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import click
import torch
import numpy as np
import scipy.sparse as ssp
from datetime import datetime
from logzero import logger, logfile
import random

from preprocess import process_ppi_pyg
from model import MultiFeatureGCN
from utils import (
    load_go_features, load_processed_features,
    load_labels, set_seed, split_dataset
)
from train import train_protein_model_multi_feature, train_with_cross_validation_multi_feature


logfile("training_detailed_1.log", maxBytes=1e9, backupCount=3)


@click.group()
def cli():
    pass


@cli.command()
@click.argument('ppi_net_mat_path', type=click.Path(exists=True))
@click.argument('pyg_graph_path', type=click.Path())
@click.argument('top', type=click.INT, default=100, required=False)
def process_ppi_cmd(ppi_net_mat_path, pyg_graph_path, top):

    process_ppi_pyg(ppi_net_mat_path, pyg_graph_path, top)


@cli.command()
@click.argument('pyg_graph_path', type=click.Path(exists=True))
@click.argument('go_features_path', type=click.Path(exists=True))
@click.argument('protein_ids_path', type=click.Path(exists=True))
@click.argument('labels_path', type=click.Path(exists=True))
@click.option('--pssm_features_path', type=click.Path(), help='Preprocessed PSSM feature file path')
@click.option('--pssm_dir', type=click.Path(), help='PSSM file directory if no preprocessed features are provided')
@click.option('--llm_features_path', type=click.Path(), help='Preprocessed LLM feature file path')
@click.option('--llm_dir', type=click.Path(), help='LLM feature file directory if no preprocessed features are provided')
@click.option('--output_dir', type=click.Path(), help='Output directory')
@click.option('--n_folds', type=click.INT, default=5, help='Number of cross-validation folds')
@click.option('--hidden_dim', type=click.INT, default=256, help='Hidden layer dimension')
@click.option('--n_layers', type=click.INT, default=2, help='Number of GCN layers')
@click.option('--epochs', type=click.INT, default=100, help='Number of training epochs')
@click.option('--lr', type=click.FLOAT, default=0.001, help='Learning rate')
@click.option('--batch_size', type=click.INT, default=64, help='Batch size')
@click.option('--weight_decay', type=click.FLOAT, default=1e-6, help='Weight decay')
@click.option('--dropout', type=click.FLOAT, default=0.3, help='Dropout rate')
@click.option('--imbalance_approach', type=click.Choice(
    ['weighted', 'undersample', 'oversample', 'mixed']),
              default='weighted', help='Method to handle imbalanced data')
@click.option('--focal_loss/--no_focal_loss', default=False, help='Use focal loss or not')
@click.option('--mixed_precision/--full_precision', default=True, help='Use mixed precision training or not')
@click.option('--seed', type=click.INT, default=42, help='Random seed')
@click.option('--split_data/--no_split_data', default=False, help='Automatically split train and test sets or not')
@click.option('--test_size', type=click.FLOAT, default=0.2, help='Test set ratio')
@click.option('--val_size', type=click.FLOAT, default=0.1, help='Validation set ratio')
@click.option('--stratify/--no_stratify', default=True, help='Use stratified sampling or not')
@click.option('--use_llm_features/--no_llm_features', default=True, help='Use LLM features or not')
def train_with_multi_feature(pyg_graph_path, go_features_path, protein_ids_path,
                             labels_path, pssm_features_path=None, pssm_dir=None,
                             llm_features_path=None, llm_dir=None, output_dir=None, **kwargs):


    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./multi_feature_contra_model_outputs_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)


    seed = kwargs.get('seed', 42)
    set_seed(seed)


    logger.info(f"Loading graph from {pyg_graph_path}...")
    pyg_graph = torch.load(pyg_graph_path)


    logger.info(f"Loading GO features from {go_features_path}...")
    go_features = load_go_features(go_features_path)
    if ssp.issparse(go_features):
        logger.info(f"GO feature matrix: {go_features.shape}, non-zero elements: {go_features.nnz}")
    else:
        logger.info(f"GO feature matrix: {go_features.shape}")


    logger.info(f"Loading protein IDs from {protein_ids_path}...")
    with open(protein_ids_path, 'r') as f:
        protein_ids = [line.strip() for line in f]

    if len(protein_ids) != go_features.shape[0]:
        raise ValueError(f"Protein ID count ({len(protein_ids)}) does not match GO feature rows ({go_features.shape[0]})")


    if pssm_features_path and os.path.exists(pssm_features_path):

        logger.info(f"Loading preprocessed PSSM features from {pssm_features_path}...")
        pssm_features = load_processed_features(pssm_features_path)
    elif pssm_dir:

        logger.info(f"Processing PSSM features from {pssm_dir}...")
        from pssm_processor import process_pssm_features
        pssm_features = process_pssm_features(pssm_dir, protein_ids)
    else:
        raise ValueError("Must provide PSSM feature path or PSSM directory")


    use_llm_features = kwargs.get('use_llm_features', True)


    if use_llm_features:
        if llm_features_path and os.path.exists(llm_features_path):

            logger.info(f"Loading preprocessed LLM features from {llm_features_path}...")
            llm_features = load_processed_features(llm_features_path)
        elif llm_dir:

            logger.info(f"Processing LLM features from {llm_dir}...")
            from llm_processor import process_llm_features
            llm_features = process_llm_features(llm_dir, protein_ids)
        else:

            logger.warning("No LLM features provided, using zero matrix instead")
            llm_features = np.zeros((go_features.shape[0], 1024))
    else:

        logger.info("LLM features not enabled, using zero matrix instead")
        llm_features = np.zeros((go_features.shape[0], 1024))


    logger.info(f"Loading labels from {labels_path}...")
    labels = load_labels(labels_path)
    logger.info(f"Labels: {labels.shape}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")


    n_folds = kwargs.get('n_folds', 5)
    hidden_dim = kwargs.get('hidden_dim', 256)
    n_layers = kwargs.get('n_layers', 2)
    epochs = kwargs.get('epochs', 100)
    lr = kwargs.get('lr', 0.001)
    batch_size = kwargs.get('batch_size', 64)
    weight_decay = kwargs.get('weight_decay', 1e-6)
    dropout = kwargs.get('dropout', 0.3)
    imbalance_approach = kwargs.get('imbalance_approach', 'weighted')
    focal_loss = kwargs.get('focal_loss', False)
    use_amp = kwargs.get('mixed_precision', True)
    split_data = kwargs.get('split_data', False)

    if split_data:

        test_size = kwargs.get('test_size', 0.2)
        val_size = kwargs.get('val_size', 0.1)
        stratify = kwargs.get('stratify', True)

        train_indices, val_indices, test_indices = split_dataset(
            labels, test_size=test_size, val_size=val_size,
            stratify=stratify, random_seed=seed
        )


        from utils import preprocess_all_features


        features_dict = preprocess_all_features(
            go_features, pssm_features, llm_features, train_indices
        )


        train_proteins = torch.LongTensor(train_indices)
        val_proteins = torch.LongTensor(val_indices)
        test_proteins = torch.LongTensor(test_indices)


        input_dims = {
            'ipr': features_dict['ipr'].shape[1],
            'pssm': features_dict['pssm'].shape[1],
            'llm': features_dict['llm'].shape[1]
        }
        output_dim = labels.shape[1]  # Number of modification types

        model = MultiFeatureGCN(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            dropout=dropout
        )


        trained_model, metrics, history = train_protein_model_multi_feature(
            model, pyg_graph, features_dict, labels,
            train_proteins, val_proteins, device,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            imbalance_approach=imbalance_approach,
            focal_loss=focal_loss, use_amp=use_amp
        )


        trained_model.eval()
        with torch.no_grad():

            features_tensor_dict = {
                'ipr': torch.FloatTensor(features_dict['ipr']).to(device),
                'pssm': torch.FloatTensor(features_dict['pssm']).to(device),
                'llm': torch.FloatTensor(features_dict['llm']).to(device)
            }

            outputs = trained_model(pyg_graph, features_tensor_dict)
            test_outputs = outputs[test_proteins]
            test_probs = torch.sigmoid(test_outputs).cpu().numpy()
            test_true = torch.FloatTensor(labels)[test_proteins].numpy()


            from eval import multi_evaluate_multilabel
            test_metrics = multi_evaluate_multilabel(test_probs, test_true)

            logger.info("\nTest set performance:")
            for k, v in test_metrics.items():
                logger.info(f"{k}: {v:.4f}")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(output_dir, f"multi_feature_gcn_model_{timestamp}.pt")

        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'input_dims': input_dims,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'n_layers': n_layers,
            'dropout': dropout,
            'go_shape': go_features.shape,
            'pssm_shape': pssm_features.shape,
            'llm_shape': llm_features.shape,
            'metrics': metrics,
            'test_metrics': test_metrics,
            'train_indices': train_indices,
            'protein_ids': protein_ids,
        }, model_path)

        logger.info(f"Model saved to {model_path}")

        return trained_model, metrics, test_metrics

    else:

        models, avg_metrics, std_metrics = train_with_cross_validation_multi_feature(
            MultiFeatureGCN, pyg_graph, go_features, pssm_features, llm_features, labels, device,
            n_folds=n_folds, epochs=epochs, lr=lr, batch_size=batch_size,
            hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout, weight_decay=weight_decay,
            imbalance_approach=imbalance_approach, focal_loss=focal_loss, use_amp=use_amp
        )


        result_file = os.path.join(output_dir, "multi_feature_results_summary.txt")
        with open(result_file, 'w') as f:
            f.write(f"Multi-feature input model cross-validation results:\n")
            f.write(f"Training parameters:\n")
            f.write(f"  - hidden_dim: {hidden_dim}\n")
            f.write(f"  - n_layers: {n_layers}\n")
            f.write(f"  - learning_rate: {lr}\n")
            f.write(f"  - weight_decay: {weight_decay}\n")
            f.write(f"  - dropout: {dropout}\n")
            f.write(f"  - batch_size: {batch_size}\n")
            f.write(f"  - epochs: {epochs}\n")
            f.write(f"  - imbalance_approach: {imbalance_approach}\n")
            f.write(f"  - focal_loss: {focal_loss}\n")
            f.write(f"  - use_llm_features: {use_llm_features}\n")
            f.write(f"  - seed: {seed}\n\n")


            metric_keys = ['roc_auc', 'pr_auc']


            for key in metric_keys:
                f.write(f"{key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}\n")

        logger.info(f"Result summary saved to {result_file}")

        return models, avg_metrics, std_metrics


if __name__ == '__main__':
    cli()