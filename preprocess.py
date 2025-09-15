#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import scipy.sparse as ssp
import torch
from torch_geometric.data import Data
from tqdm import tqdm, trange
from logzero import logger

from utils import get_norm_net_mat


def process_ppi_pyg(ppi_net_mat_path, pyg_graph_path, top):


    start_time = time.time()
    logger.info(f"Loading PPI network from {ppi_net_mat_path}...")
    ppi_net_mat = ssp.load_npz(ppi_net_mat_path)


    ppi_net_mat = ppi_net_mat + ssp.eye(ppi_net_mat.shape[0], format='csr')
    logger.info(f'PPI network: {ppi_net_mat.shape}, {ppi_net_mat.nnz} edges')

    logger.info(f"Processing top {top} connections...")
    r, c, v = [], [], []
    for i in trange(ppi_net_mat.shape[0]):
        if len(ppi_net_mat[i].data) > 0:
            for v_, c_ in sorted(zip(ppi_net_mat[i].data, ppi_net_mat[i].indices), reverse=True)[:top]:
                r.append(i)
                c.append(c_)
                v.append(v_)

    logger.info("Normalizing network matrix...")
    ppi_net_mat = get_norm_net_mat(ssp.csc_matrix((v, (r, c)), shape=ppi_net_mat.shape).T)
    logger.info(f'Processed PPI network: {ppi_net_mat.shape}, {ppi_net_mat.nnz} edges')


    ppi_net_mat_coo = ssp.coo_matrix(ppi_net_mat)


    edge_index = torch.tensor([ppi_net_mat_coo.row, ppi_net_mat_coo.col], dtype=torch.long)


    edge_attr = torch.tensor(ppi_net_mat_coo.data, dtype=torch.float).view(-1, 1)


    pyg_data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=ppi_net_mat.shape[0])


    max_in_degree = 0
    node_degrees = torch.bincount(edge_index[1])
    if len(node_degrees) > 0:
        max_in_degree = node_degrees.max().item()
    assert max_in_degree <= top, f"Maximum in-degree ({max_in_degree}) exceeds specified value ({top})"

    logger.info(f"Saving graph to {pyg_graph_path}...")
    torch.save(pyg_data, pyg_graph_path)

    process_time = time.time() - start_time
    logger.info(f"PPI processing completed in {process_time:.2f} seconds")

    return pyg_data