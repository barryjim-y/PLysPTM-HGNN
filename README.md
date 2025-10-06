# PLysPTM-HGNN
PLysPTM-HGNN is designed for the prediction of protein lysine post-translational modification (PTM) types. It integrates Gene Ontology (GO) features, Position-Specific Scoring Matrix (PSSM) features, and Large Language Model (LLM) embeddings. These features are seperately processed by a linear transformation, hybrid graph neural network, and convolutional neural network combiner. Then, the refined features are concatenated and fed into a fully connected layer for makeing predictions.
![PLysPTM-HGNN Framework](images/Figure%202.pdf)


# Requirements
```
python 3.9
torch >= 1.13
torch_geometric >= 2.3
numpy >= 1.21
scipy >= 1.8
scikit-learn >= 1.0
logzero
tqdm
click
```

# Usage
```bash
git clone https://github.com/barryjim-y/PLysPTM-HGNN
cd PLysPTM-HGNN

# Step 1. Preprocess PPI network
python main.py process-ppi-cmd matrix.npz graph.bin 100

# Step 2. Train model with multi-feature input (example script)
python main.py train-with-multi-feature graph_1.bin feature_matrix.npz ProteinList.txt lysine_matrix.npy \
    --pssm_features_path data/pssm_features.npy \
    --llm_features_path data/llm_features.npy \
    --n_folds 10 --hidden_dim 254 --n_layers 3 --epochs 200 \
    --lr 0.001 --batch_size 128 --weight_decay 1e-3 --dropout 0.3 \
    --imbalance_approach mixed --no_focal_loss --seed 42
```

# Directory Structure Overview

`main.py`  
Entry point for preprocessing, training, and evaluation.  

`model.py`  
Definition of ProtLysMGCN and ablation variants.  

`train.py`  
Training functions, cross-validation, and performance logging.  

`eval.py`  
Evaluation metrics (ROC-AUC, PR-AUC).  

`preprocess.py`  
Builds PyTorch Geometric graphs from PPI networks.  

`utils.py`  
Helper functions for features, labels, and imbalance handling.  
