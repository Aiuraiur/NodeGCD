"""Data loading, graph splitting and evaluation utilities."""
from typing import AnyStr
import random
import numpy as np
import copy
import torch
import dgl
import dgl.data as dgl_data
from ogb.nodeproppred import DglNodePropPredDataset

from scipy.optimize import linear_sum_assignment


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def getDataset(dataset_name: AnyStr):
    if dataset_name == "cora":
        dataset = dgl_data.CoraGraphDataset(verbose=False)[0]
    elif dataset_name == "citeseer":
        dataset = dgl_data.CiteseerGraphDataset(verbose=False)[0]
    elif dataset_name == "amazon_photos":
        dataset = dgl_data.AmazonCoBuyPhotoDataset(verbose=False)[0]
    elif dataset_name == "amazon_computers":
        dataset = dgl_data.AmazonCoBuyComputerDataset(verbose=False)[0]
    elif dataset_name == "coauthor_cs":
        dataset = dgl_data.CoauthorCSDataset(verbose=False)[0]
    elif dataset_name == "wikics":
        dataset = dgl_data.WikiCSDataset(verbose=False)[0]
    elif dataset_name == "arxiv":
        data = DglNodePropPredDataset(name="ogbn-arxiv", root='data/')
        g, labels = data[0]
        g.ndata["label"] = labels.squeeze()
        dataset = g
    else:
        raise ValueError("Unknown dataset")
    return dataset


def split_node_dataset(dataset: dgl.DGLGraph, seed: int,
                       train_ratio=0.1, val_ratio=0.1):
    """Split nodes into seen / novel classes (50% / 50%).

    Seen classes: 10% train, 10% val, 80% test.
    Novel classes: all nodes in test set.
    """
    setup_seed(seed)
    g = dataset
    num_nodes = g.num_nodes()
    labels = g.ndata["label"].numpy()
    n_class = len(set(labels))
    class_list = list(range(n_class))
    n_train_class = round(n_class / 2)
    seen_classes = class_list[:n_train_class]
    novel_classes = class_list[n_train_class:]

    print(f"Seen classes: {seen_classes}")
    print(f"Novel classes: {novel_classes}")

    train_idx, val_idx, test_idx = [], [], []

    for cls_id in seen_classes:
        candidate = np.where(labels == cls_id)[0].tolist()
        random.shuffle(candidate)
        n_total = len(candidate)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        train_idx.extend(candidate[:n_train])
        val_idx.extend(candidate[n_train:n_train + n_val])
        test_idx.extend(candidate[n_train + n_val:])

    for cls_id in novel_classes:
        test_idx.extend(np.where(labels == cls_id)[0])

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = True
        return mask

    g.ndata["train_mask"] = get_mask(np.array(train_idx))
    g.ndata["val_mask"] = get_mask(np.array(val_idx))
    g.ndata["test_mask"] = get_mask(np.array(test_idx))

    mask_lab = (g.ndata["train_mask"] | g.ndata["val_mask"]).numpy()
    mask_cls = np.array([x in seen_classes for x in labels])

    return mask_lab, mask_cls, n_class, n_train_class


def load_data(dataset_str: str, seed: int):
    g = getDataset(dataset_str)
    g = dgl.add_self_loop(g)

    mask_lab, mask_cls, n_class, n_train_class = split_node_dataset(g, seed)

    input_dim = g.ndata["feat"].shape[1]

    print("Number of nodes:", g.num_nodes())
    print("Number of edges:", g.num_edges())
    print("Input dim:", input_dim)
    print("Num classes:", n_class)
    print("Seen classes:", n_train_class)

    return g, input_dim, n_class, n_train_class, mask_lab, mask_cls


def cluster_acc(y_pred, y_true):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def split_cluster_acc_v2(y_true, y_pred, mask):
    """Cluster accuracy split into seen / novel classes."""
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.vstack(linear_sum_assignment(w.max() - w)).T
    ind_map = {j: i for i, j in ind}

    total_acc = sum([w[i, j] for i, j in ind]) / y_pred.size

    old_acc, total_old_instances = 0, 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc, total_new_instances = 0, 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    if total_new_instances != 0:
        new_acc /= total_new_instances

    return total_acc, old_acc, new_acc


def drop_edge(graph, drop_rate=0.2):
    g = copy.deepcopy(graph)
    num_edges = g.num_edges()
    mask = torch.rand(num_edges) > drop_rate
    src, dst = g.edges()
    src, dst = src[mask], dst[mask]
    new_g = dgl.graph((src, dst), num_nodes=g.num_nodes())
    new_g = dgl.add_self_loop(new_g)
    for key in g.ndata:
        new_g.ndata[key] = g.ndata[key]
    return new_g


def mask_feature(x, mask_rate=0.2):
    x = x.clone()
    mask = torch.rand_like(x) < mask_rate
    x[mask] = 0
    return x
