"""
Node GCD: Generalized Category Discovery on Graphs.

Unified training script: select a dataset with `--dataset` and an
experimental variant with `--variant`.

Variants:
  A  : multi-prototype re-initialization after every K prediction with
       progressive merging (joint denominator)
  A1 : A + per-view averaged loss (separate_view=True)
  A2 : A1 + q temperature annealing (tau_q = proto_tau * q_gamma^epoch)
  A3 : A1 + marginal entropy regularization (anti-collapse)
  B  : multi-prototype only during the first multi_proto_rounds K
       predictions, single prototype afterwards
  B1 : B + per-view averaged loss (DEFAULT variant). On arxiv, the
       arxiv-specific configuration (auxiliary CE losses + class-center
       diversity + hidden 256, see ARXIV_B1_OVERRIDES) is applied
       automatically.

Key design choices:
  proto_init=class_kmeans : class-aware prototype initialization (seen
       classes: KMeans within labeled nodes per class; novel classes:
       global KMeans + Hungarian alignment, then per-cluster KMeans)
  mask_aware (default True): active-prototype masks enter N2P/P2C losses,
       so merged-away prototypes no longer contribute positives/negatives
  estimate_k=False : fixed-K training with ground-truth class count
"""
import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows")
import argparse
import numpy as np
import os
import sys

import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn import metrics

# Add project root to path so the script can be run from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.model import GNNModel
from scipy.optimize import linear_sum_assignment, minimize_scalar
import scipy.sparse as sp
from losses.supcon import SupConLoss
from losses.n2p_loss import N2PContrastiveLoss
from losses.p2c_loss import P2CContrastiveLoss
from networks.prototype import ProtoClassifier
from networks.prototype_manager import PrototypeManager
from utils import *


# Default configuration per dataset (best historical B1 settings)
DATASET_CONFIGS = {
    "cora":    dict(num_epochs=1000, hidden_dim=128, num_gnn_heads=4, n_prototypes=10,
                    warmup_epochs=10, edge_drop_rate=0.2, feat_mask_rate=0.2),
    "citeseer": dict(num_epochs=500, hidden_dim=128, num_gnn_heads=4, n_prototypes=5,
                     warmup_epochs=10, edge_drop_rate=0.2, feat_mask_rate=0.2),
    "amazon_photos": dict(num_epochs=1000, hidden_dim=32, num_gnn_heads=4, n_prototypes=10,
                          warmup_epochs=10, edge_drop_rate=0.5, feat_mask_rate=0.5,
                          weight_decay=1e-6),
    "amazon_computers": dict(num_epochs=1000, hidden_dim=128, num_gnn_heads=2, n_prototypes=10,
                             warmup_epochs=5, edge_drop_rate=0.1, feat_mask_rate=0.1),
    "coauthor_cs": dict(num_epochs=1000, hidden_dim=128, num_gnn_heads=4, n_prototypes=10,
                        warmup_epochs=10, edge_drop_rate=0.2, feat_mask_rate=0.2),
    "wikics": dict(num_epochs=1000, hidden_dim=128, num_gnn_heads=4, n_prototypes=10,
                   warmup_epochs=10, edge_drop_rate=0.2, feat_mask_rate=0.2),
    "arxiv": dict(num_epochs=1000, hidden_dim=128, num_gnn_heads=4, n_prototypes=10,
                  warmup_epochs=10, edge_drop_rate=0.5, feat_mask_rate=0.5,
                  max_novel_k=30),
}

# Historical arxiv_B1 configuration for reproducing reported results.
# Contains the full loss stack: supervised CE + prototype CE + class-center
# diversity + fixed P2C balancing. Small-graph B1 scripts do not include
# these components, so they are disabled by default and only forced on for
# arxiv_B1.
ARXIV_B1_OVERRIDES = dict(
    hidden_dim=256, lr=0.003, weight_decay=1e-4, proto_tau=0.5,
    n_prototypes=10, ema_momentum=0.5, p2c_weight=2.0, proto_init="labeled",
    feat_drop_rate=0.3, attn_drop_rate=0.3,
    ce_weight=0.5, proto_ce_weight=1.0, center_div_weight=1.0,
    mask_aware=False, p2c_balance_mode="fixed",
)

# Historical per-(dataset, variant) default corrections. Applied only when
# the current value equals the parser default or the dataset config value
# (explicit CLI arguments always win).
VARIANT_OVERRIDES = {
    ("amazon_photos", "A1"): dict(num_epochs=500, hidden_dim=128, n_prototypes=5,
                                   weight_decay=1e-5, edge_drop_rate=0.2,
                                   feat_mask_rate=0.2),
    ("amazon_computers", "A1"): dict(num_epochs=500, num_gnn_heads=4, n_prototypes=5,
                                      warmup_epochs=10, edge_drop_rate=0.2,
                                      feat_mask_rate=0.2),
    ("cora", "A"):  dict(num_epochs=500, n_prototypes=5),
    ("cora", "A2"): dict(num_epochs=500, n_prototypes=5),
    ("cora", "A3"): dict(num_epochs=500, n_prototypes=5),
    ("cora", "B"):  dict(num_epochs=500, n_prototypes=5),
}

# Variant behavior
VARIANT_CONFIGS = {
    "A":  dict(separate_view=False, q_gamma=None, entropy_weight=0.0, multi_proto_mode="always"),
    "A1": dict(separate_view=True,  q_gamma=None, entropy_weight=0.0, multi_proto_mode="always"),
    "A2": dict(separate_view=True,  q_gamma=0.995, entropy_weight=0.0, multi_proto_mode="always"),
    "A3": dict(separate_view=True,  q_gamma=None, entropy_weight=0.5, multi_proto_mode="always"),
    "B":  dict(separate_view=False, q_gamma=None, entropy_weight=0.0, multi_proto_mode="phased"),
    "B1": dict(separate_view=True,  q_gamma=None, entropy_weight=0.0, multi_proto_mode="phased"),
}

ALL_DATASETS = list(DATASET_CONFIGS.keys())
ALL_VARIANTS = list(VARIANT_CONFIGS.keys())


def parse_args(dataset_override=None, variant_override=None):
    parser = argparse.ArgumentParser(
        description="Node GCD training script (--dataset + --variant)")
    # dataset and variant
    parser.add_argument("--dataset", type=str,
                        default=dataset_override or "cora",
                        choices=ALL_DATASETS)
    parser.add_argument("--variant", type=str,
                        default=variant_override or "B1",
                        choices=ALL_VARIANTS,
                        help="experiment variant: A/A1/A2/A3/B/B1 (default B1)")
    parser.add_argument("--no_arxiv_b1_override", action='store_true', default=False,
                        help="disable the arxiv_B1 config override (ablation: pass "
                             "all parameters explicitly)")
    # basic training parameters
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="training epochs after warmup (default: dataset config)")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    # GNN encoder parameters
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="hidden dimension (default: dataset config)")
    parser.add_argument("--num_gnn_layers", type=int, default=2)
    parser.add_argument("--num_gnn_heads", type=int, default=None,
                        help="number of GAT heads (default: dataset config)")
    parser.add_argument("--feat_drop_rate", type=float, default=0.2)
    parser.add_argument("--attn_drop_rate", type=float, default=0.2)
    parser.add_argument("--encoder_name", type=str, default="gat")
    # prototype learning parameters
    parser.add_argument('--gamma', type=float, default=1.0,
                        help="N2P temperature annealing factor (baseline 1.0 = off)")
    parser.add_argument("--proto_tau", type=float, default=0.3,
                        help="shared temperature for N2P and P2C")
    parser.add_argument("--n_prototypes", type=int, default=None,
                        help="prototypes per class (default: dataset config)")
    parser.add_argument("--use_weight", action='store_true', default=True)
    parser.add_argument("--no_weight", action='store_false', dest='use_weight')
    parser.add_argument("--warmup_epochs", type=int, default=None,
                        help="SupCon pretraining epochs (default: dataset config)")
    parser.add_argument("--supcon_warmup", action='store_true', default=True)
    parser.add_argument("--no_supcon_warmup", action='store_false', dest='supcon_warmup',
                        help="disable SupCon pretraining")
    parser.add_argument("--supcon_tau", type=float, default=0.07)
    parser.add_argument("--warmup_max_nodes", type=int, default=4000,
                        help="max nodes for warmup SupCon (0 = full graph; "
                             "mini-batch size on large graphs)")
    parser.add_argument("--ema_momentum", type=float, default=1.0,
                        help="EMA coefficient for class-level prototypes "
                             "(baseline 1.0 = off)")
    parser.add_argument("--p2c_weight", type=float, default=1.0)
    parser.add_argument("--disable_p2c", action='store_true', default=False)
    parser.add_argument("--p2c_tau", type=float, default=0.1,
                        help="logit temperature for P2C / prototype-CE / center-CE")
    parser.add_argument("--p2c_balance_mode", type=str, default="adaptive",
                        choices=["adaptive", "fixed"],
                        help="P2C balancing: adaptive = scaled by N2P/P2C ratio; "
                             "fixed = constant p2c_weight")
    # supervised CE and regularization losses (default off for small graphs)
    parser.add_argument("--use_linear_cls", action='store_true', default=True,
                        help="create the linear classification head when ce_weight > 0")
    parser.add_argument("--no_linear_cls", action='store_false', dest='use_linear_cls')
    parser.add_argument("--ce_weight", type=float, default=0.0,
                        help="linear-head CE weight (seen ground-truth labels only)")
    parser.add_argument("--proto_ce_weight", type=float, default=0.0,
                        help="prototype-guided CE weight (gradients flow into prototypes)")
    parser.add_argument("--center_ce_weight", type=float, default=0.0,
                        help="EMA class-center guided CE weight (gradients flow "
                             "into the encoder)")
    parser.add_argument("--proto_div_weight", type=float, default=0.0,
                        help="intra-class prototype diversity regularization weight")
    parser.add_argument("--center_div_weight", type=float, default=0.0,
                        help="inter-class center diversity regularization weight")
    parser.add_argument("--entropy_weight", type=float, default=0.0,
                        help="marginal entropy maximization weight "
                             "(L_entropy = -H(p), anti-collapse; A3 variant = 0.5)")
    parser.add_argument("--gat_view", type=str, default='head', choices=['head', 'layer'])
    # structural choices
    parser.add_argument("--w_same_view", action='store_true', default=False)
    parser.add_argument("--w_cross_view", action='store_false', dest='w_same_view')
    parser.add_argument("--hard_count", action='store_true', default=True,
                        help="update alpha with hard (argmax) counts")
    parser.add_argument("--soft_count", action='store_false', dest='hard_count')
    # K estimation parameters
    parser.add_argument("--estimate_k", action='store_true', default=True,
                        help="dynamically estimate the number of novel classes "
                             "(False = fixed-K experiment)")
    parser.add_argument("--no_estimate_k", action='store_false', dest='estimate_k')
    parser.add_argument("--max_novel_k", type=int, default=10,
                        help="upper bound of the novel class count (arxiv default 30)")
    parser.add_argument("--k_score_weights", type=str, default="1,1,1",
                        help="K estimation score weights: "
                             "modularity,separation,labeled_acc[,silhouette]")
    parser.add_argument("--k_refine_rounds", type=int, default=10)
    parser.add_argument("--multi_proto_rounds", type=int, default=-1,
                        help="multi-prototype K prediction rounds, -1 = half of rounds")
    parser.add_argument("--merge_interval", type=int, default=10,
                        help="progressive merge interval (epochs)")
    # prototype initialization
    parser.add_argument("--proto_init", type=str, default="kmeans",
                        choices=["random", "kmeans", "labeled", "class_kmeans"],
                        help="random/kmeans/labeled/class_kmeans (class-aware)")
    # mask-aware losses
    parser.add_argument("--mask_aware", action='store_true', default=True,
                        help="exclude inactive prototypes from N2P/P2C losses")
    parser.add_argument("--no_mask_aware", action='store_false', dest='mask_aware')
    # merge mode
    parser.add_argument("--merge_mode", type=str, default="alpha",
                        choices=["alpha", "w_centroid"],
                        help="alpha = alpha-weighted prototype-vector merge (default); "
                             "w_centroid = union of w-weighted node centroids")
    # fixed-K: whether to run one multi-prototype init + progressive merging
    parser.add_argument("--merge_when_fixed_k", action='store_true', default=True)
    parser.add_argument("--no_merge_when_fixed_k", action='store_false',
                        dest='merge_when_fixed_k',
                        help="fixed-K without progressive merging (M stays constant)")
    # q computation: decouple temperatures of w and exp (off by default)
    parser.add_argument("--decouple_q_tau", action='store_true', default=False,
                        help="use different temperatures for w and exp in q")
    # data augmentation
    parser.add_argument("--edge_drop_rate", type=float, default=None,
                        help="edge drop rate (default: dataset config)")
    parser.add_argument("--feat_mask_rate", type=float, default=None,
                        help="feature mask rate (default: dataset config)")
    # plotting helpers
    parser.add_argument("--single_run", action="store_true", default=False)
    parser.add_argument("--epochs_csv", type=str, default="",
                        help="path for per-epoch metric CSV output")
    parser.add_argument("--eval_freq", type=int, default=1,
                        help="evaluation/CSV frequency (every N epochs + last)")
    parser.add_argument("--eval_proto_init", action='store_true', default=False,
                        help="warm-start evaluation KMeans with class centers")

    args = parser.parse_args()

    # Apply dataset default config (explicit CLI arguments win)
    cfg = DATASET_CONFIGS[args.dataset]
    for key in ("num_epochs", "hidden_dim", "num_gnn_heads", "n_prototypes",
                "warmup_epochs", "edge_drop_rate", "feat_mask_rate"):
        if getattr(args, key) is None:
            setattr(args, key, cfg[key])
    if args.max_novel_k is None or args.dataset == "arxiv" and args.max_novel_k == 10:
        # arxiv defaults to 30, other datasets to 10
        args.max_novel_k = cfg.get("max_novel_k", 10)
    for key in ("lr", "weight_decay"):
        if key in cfg and getattr(args, key) == parser.get_default(key):
            setattr(args, key, cfg[key])

    # Historical arxiv configuration reproduction. Applied automatically on
    # arxiv (full loss stack: supervised CE + prototype CE + class-center
    # diversity + fixed P2C balancing + hidden 256). Disable with
    # --no_arxiv_b1_override for ablation.
    if args.dataset == "arxiv" and not args.no_arxiv_b1_override:
        for key, val in ARXIV_B1_OVERRIDES.items():
            if getattr(args, key) == parser.get_default(key) or key in (
                    "hidden_dim", "n_prototypes"):
                setattr(args, key, val)

    # Historical per-(dataset, variant) corrections
    vov = VARIANT_OVERRIDES.get((args.dataset, args.variant), {})
    for key, val in vov.items():
        cur = getattr(args, key)
        if cur == parser.get_default(key) or (key in cfg and cur == cfg[key]):
            setattr(args, key, val)

    # Variant behavior
    vcfg = VARIANT_CONFIGS[args.variant]
    args.variant_separate_view = vcfg["separate_view"]
    args.variant_q_gamma = vcfg["q_gamma"]
    args.variant_entropy_weight = vcfg["entropy_weight"]
    args.variant_multi_proto_mode = vcfg["multi_proto_mode"]
    if args.variant_q_gamma is not None:
        args.q_gamma = args.variant_q_gamma
    # variant entropy weight only applies when not set explicitly via CLI
    if args.variant_entropy_weight > 0 and \
            args.entropy_weight == parser.get_default("entropy_weight"):
        args.entropy_weight = args.variant_entropy_weight

    # q temperature annealing (A2) / marginal entropy (A3) weights
    if not hasattr(args, "q_gamma"):
        args.q_gamma = None
    if not hasattr(args, "entropy_weight"):
        args.entropy_weight = 0.0

    return args


# ==========================================================
# Evaluation + geometry metrics (IR/SR)
# ==========================================================
@torch.no_grad()
def evaluate(model, g, labels, n_class, mask_cls, device, proto_init=None):
    """KMeans + Hungarian evaluation, also returns geometry metrics.

    proto_init: [V, C, D] learned class centers; when given, warm-starts
        KMeans (n_init=1); otherwise plain random KMeans.
    """
    model.eval()
    x = g.ndata["feat"].float().to(device)
    emb = model(x, g)  # [N, D]
    if proto_init is not None:
        init_centroids = proto_init.mean(dim=0).cpu().numpy()  # [C, D]
        if init_centroids.shape[0] != n_class:
            init_centroids = init_centroids[:n_class]
        kmeans = KMeans(n_clusters=n_class, init=init_centroids, n_init=1,
                        random_state=0)
    else:
        kmeans = KMeans(n_clusters=n_class, random_state=0)
    pred_all = kmeans.fit_predict(emb.cpu().numpy())

    test_mask = g.ndata["test_mask"].cpu().numpy()
    test_labels = labels[test_mask].cpu().numpy()
    test_pred = pred_all[test_mask]
    mask_cls_test = mask_cls[test_mask]
    all_acc, old_acc, new_acc = split_cluster_acc_v2(test_labels, test_pred, mask_cls_test)

    geometry = compute_geometry_metrics(emb, labels, mask_cls)
    model.train()
    return all_acc, old_acc, new_acc, geometry


@torch.no_grad()
def compute_geometry_metrics(emb, labels, mask_cls):
    """Seen/Novel intra-class variance + Imbalance Rate + Separation Rate.

    emb:  [N, D] (torch)
    labels: [N] (torch)
    mask_cls: [N] bool numpy (True = seen-class nodes)
    """
    e = emb.detach().cpu().numpy()
    lab = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    mask = mask_cls.astype(bool)

    seen_idx = np.where(mask)[0]
    novel_idx = np.where(~mask)[0]

    def _var(idx):
        if len(idx) <= 1:
            return 0.0
        mu = e[idx].mean(axis=0, keepdims=True)
        return float(((e[idx] - mu) ** 2).sum() / len(idx))

    seen_var = _var(seen_idx)
    novel_var = _var(novel_idx)

    if seen_var + novel_var < 1e-12:
        ir, sr = 1.0, 0.0
    else:
        ir = max(seen_var, novel_var) / (min(seen_var, novel_var) + 1e-12)
        mu_seen = e[seen_idx].mean(axis=0)
        mu_novel = e[novel_idx].mean(axis=0)
        sr = float(np.linalg.norm(mu_seen - mu_novel)
                   / (np.sqrt(seen_var) + np.sqrt(novel_var) + 1e-12))

    return {"seen_var": seen_var, "novel_var": novel_var, "IR": ir, "SR": sr}


# ==========================================================
# Build multi-views (head-based / layer-based)
# ==========================================================
def build_views(model, g, x, edge_drop_rate, feat_mask_rate, device, gat_view='head'):
    if gat_view != 'aug':
        gat_views = model(x, g, view_type=gat_view)  # [N, V, D]
        views = gat_views.permute(1, 0, 2)            # [V, N, D]
        return views

    g1 = drop_edge(g, edge_drop_rate).to(device)
    g2 = drop_edge(g, edge_drop_rate).to(device)
    x1 = mask_feature(x, feat_mask_rate)
    x2 = mask_feature(x, feat_mask_rate)
    view1 = model(x1, g1)
    view2 = model(x2, g2)
    views = torch.stack([view1, view2], dim=0)
    return views


# ==========================================================
# Prototype assignment (w -> q -> pseudo labels)
# ==========================================================
def compute_prototype_assignment(
        classifier, manager, views, n_class, n_prototypes, hidden_dim,
        n_train_class, tau, only_seen=False, decouple_q_tau=False,
        chunk_size=16384,
):
    """Prototype assignment (w -> q -> pseudo labels).

    Chunked over the node dimension to bound GPU memory on large graphs:
    sim/w/exp are materialized per chunk and alpha/centroid statistics are
    accumulated within the loop. Equivalence: q normalization and the w
    softmax are per-(n,v) independent; alpha via sum over (n,v) differs
    from the mean only by the constant N*V and cancels after normalization.
    For N <= chunk_size the single-chunk loop matches the unchunked
    implementation bitwise. Returns w_stats instead of the full w tensor:
    alpha_sum/hard_cnt for the alpha update, num_w/den_w for the
    w-weighted node centroids.
    """
    prototypes = classifier.get_prototypes()
    n_views = prototypes.shape[0]
    proto_reshape = prototypes.view(n_views, n_class, n_prototypes, hidden_dim)

    views_norm = F.normalize(views, dim=-1)

    if only_seen:
        proto_norm = F.normalize(proto_reshape[:, :n_train_class, :, :], dim=-1)
        C_eff = n_train_class
    else:
        proto_norm = F.normalize(proto_reshape, dim=-1)
        C_eff = n_class

    mask = manager.prototype_mask.unsqueeze(0).unsqueeze(0)
    if only_seen:
        mask = mask[:, :, :n_train_class, :]

    # temperatures of w and exp (same by default; decouple_q_tau uses a
    # larger temperature for w)
    tau_w = tau
    if decouple_q_tau:
        tau_w = tau * 3.0

    N = views_norm.shape[1]
    chunk = min(chunk_size, N)

    q_parts = []
    alpha_sum = torch.zeros(C_eff, n_prototypes, device=views.device)
    hard_cnt = torch.zeros(C_eff, n_prototypes, device=views.device)
    num_w = torch.zeros(n_views, C_eff, n_prototypes, hidden_dim, device=views.device)
    den_w = torch.zeros(n_views, C_eff, n_prototypes, device=views.device)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        v_chunk = views_norm[:, start:end]                       # [V, B, D]
        sim_chunk = torch.einsum('vnd,vckd->nvck', v_chunk, proto_norm)
        sim_chunk = sim_chunk.masked_fill(mask == 0, -1e9)

        w_chunk = F.softmax(sim_chunk / tau_w, dim=-1)
        score = (w_chunk * torch.exp(sim_chunk / tau)).sum(-1)

        q_chunk = score / (score.sum(dim=-1, keepdim=True) + 1e-12)
        q_parts.append(q_chunk.mean(dim=1))                      # [B, C]

        # alpha statistics: hard counts (argmax) + soft counts (w sums)
        proto_assign = w_chunk.argmax(dim=-1)                    # [B, V, C]
        one_hot = F.one_hot(proto_assign, num_classes=n_prototypes).float()
        hard_cnt += one_hot.sum(dim=(0, 1))                      # [C, K]
        alpha_sum += w_chunk.detach().sum(dim=(0, 1))            # [C, K]

        # w-weighted node centroids (for w_centroid merge + EMA class
        # centers). The statistics are consumed by no-grad EMA updates,
        # so detach does not change values or training dynamics while
        # avoiding accumulating the graph over chunks.
        num_w += torch.einsum('nvck,vnd->vckd', w_chunk.detach(), v_chunk)
        den_w += w_chunk.detach().sum(dim=0)                     # [V, C, K]

    q = torch.cat(q_parts, dim=0)                                # [N, C]
    w_stats = {"alpha_sum": alpha_sum, "hard_cnt": hard_cnt,
               "num_w": num_w, "den_w": den_w}

    if only_seen:
        q_full = torch.zeros(N, n_class, device=views.device)
        q_full[:, :n_train_class] = q
        pseudo_labels = manager.get_pseudo_labels(q)
        return prototypes, proto_reshape, q_full, pseudo_labels, w_stats
    else:
        pseudo_labels = manager.get_pseudo_labels(q)
        return prototypes, proto_reshape, q, pseudo_labels, w_stats


# ==========================================================
# Class-aware prototype initialization
#   seen classes : KMeans within labeled nodes of each class -> M sub-prototypes
#   novel classes: global KMeans -> Hungarian alignment to seen classes
#                  -> unmatched clusters are novel -> per-cluster KMeans
# ==========================================================
@torch.no_grad()
def _class_aware_centers(emb_np, labels_np, mask_lab_np, n_train_class,
                         n_class_total, n_prototypes, seed):
    """Return class-aware prototype matrix of shape [n_class_total * M, D]."""
    D = emb_np.shape[1]
    centers = np.zeros((n_class_total, n_prototypes, D), dtype=np.float32)

    # ---- seen classes: KMeans within labeled nodes ----
    for c in range(n_train_class):
        idx = np.where((labels_np == c) & mask_lab_np)[0]
        if len(idx) == 0:
            centers[c] = np.random.randn(n_prototypes, D).astype(np.float32)
            continue
        if len(idx) < n_prototypes:
            # too few nodes: use the nodes themselves + noise to fill
            base = emb_np[idx]
            centers[c, :len(idx)] = base
            noise = np.random.randn(n_prototypes - len(idx), D).astype(np.float32) * 0.01
            centers[c, len(idx):] = emb_np[idx].mean(axis=0, keepdims=True) + noise
        else:
            km = KMeans(n_clusters=n_prototypes, random_state=seed, n_init=10)
            km.fit(emb_np[idx])
            centers[c] = km.cluster_centers_

    # ---- novel classes: global KMeans -> alignment -> per-cluster KMeans ----
    n_novel = n_class_total - n_train_class
    if n_novel > 0:
        km_global = KMeans(n_clusters=n_class_total, random_state=seed, n_init=10)
        global_labels = km_global.fit_predict(emb_np)

        # Hungarian: clusters -> seen classes (via labeled nodes)
        lab_idx = np.where(mask_lab_np)[0]
        w_mat = np.zeros((n_class_total, n_class_total), dtype=np.int64)
        for i in lab_idx:
            w_mat[global_labels[i], labels_np[i]] += 1
        row_ind, col_ind = linear_sum_assignment(w_mat.max() - w_mat)
        matched_clusters = set(row_ind.tolist())

        novel_slot = 0
        for cl in range(n_class_total):
            if cl in matched_clusters:
                continue  # cluster corresponds to a seen class
            if novel_slot >= n_novel:
                break
            c = n_train_class + novel_slot
            idx = np.where(global_labels == cl)[0]
            if len(idx) >= n_prototypes:
                km = KMeans(n_clusters=n_prototypes, random_state=seed, n_init=10)
                km.fit(emb_np[idx])
                centers[c] = km.cluster_centers_
            elif len(idx) > 0:
                centers[c, :len(idx)] = emb_np[idx]
                noise = np.random.randn(n_prototypes - len(idx), D).astype(np.float32) * 0.01
                centers[c, len(idx):] = emb_np[idx].mean(axis=0, keepdims=True) + noise
            else:
                centers[c] = np.random.randn(n_prototypes, D).astype(np.float32)
            novel_slot += 1
        # fill remaining novel slots from global KMeans centers
        while novel_slot < n_novel:
            c = n_train_class + novel_slot
            centers[c] = km_global.cluster_centers_[
                novel_slot % len(km_global.cluster_centers_)]
            novel_slot += 1

    return centers.reshape(n_class_total * n_prototypes, D)


@torch.no_grad()
def reinit_prototypes_multi(
        classifier, manager, model, g, x, labels_np, mask_lab_np,
        n_train_class, n_class, n_prototypes, n_views, hidden_dim, seed, device,
        proto_init="kmeans",
):
    """Warm-start multi-prototype re-initialization after a K prediction.

    proto_init:
      kmeans       : global KMeans(C*M) (legacy behavior)
      class_kmeans : class-aware initialization
      labeled      : seen classes from labeled means + noise, novel random
    """
    model.eval()
    emb_np = model(x, g).detach().cpu().numpy()
    model.train()

    n_total = n_class * n_prototypes
    if n_total > len(emb_np):
        n_total = len(emb_np)

    if proto_init == "class_kmeans":
        centers_np = _class_aware_centers(
            emb_np, labels_np, mask_lab_np, n_train_class,
            n_class_total=n_class, n_prototypes=n_prototypes, seed=seed)[:n_total]
    elif proto_init == "labeled":
        centers_np = _labeled_centers(
            emb_np, labels_np, mask_lab_np, n_train_class, n_class, n_prototypes)[:n_total]
    else:
        kmeans = KMeans(n_clusters=n_total, random_state=seed, n_init=10)
        kmeans.fit(emb_np)
        centers_np = kmeans.cluster_centers_
        print(f"  [Reinit-Multi] KMeans inertia={kmeans.inertia_:.1f}")

    centers = torch.from_numpy(centers_np).float().to(device)
    centers_norm = F.normalize(centers, dim=-1)

    for v in range(n_views):
        copy_len = min(n_total, classifier.prototypes.data[v].shape[0])
        classifier.prototypes.data[v, :copy_len] = centers_norm[:copy_len]

    # reset mask and alpha (all n_prototypes active)
    manager.prototype_mask[:] = 1
    manager.alpha[:] = 1.0 / n_prototypes

    print(f"  [Reinit-Multi] n_class={n_class}, n_proto={n_prototypes}, mode={proto_init}")


@torch.no_grad()
def _labeled_centers(emb_np, labels_np, mask_lab_np, n_train_class,
                     n_class_total, n_prototypes):
    """labeled mode: seen classes from labeled means + noise, novel random."""
    D = emb_np.shape[1]
    out = np.zeros((n_class_total * n_prototypes, D), dtype=np.float32)
    for c in range(n_class_total):
        if c < n_train_class:
            idx = np.where((labels_np == c) & mask_lab_np)[0]
            centroid = emb_np[idx].mean(axis=0) if len(idx) > 0 else np.zeros(D)
            for k in range(n_prototypes):
                noise = np.random.randn(D).astype(np.float32) * 0.01
                out[c * n_prototypes + k] = centroid + noise
        else:
            for k in range(n_prototypes):
                out[c * n_prototypes + k] = np.random.randn(D).astype(np.float32)
    return out


@torch.no_grad()
def reinit_prototypes_single(
        classifier, manager, model, g, x, labels_np, mask_lab_np,
        n_class, n_prototypes_max, n_views, hidden_dim, seed, device,
        proto_init="kmeans", n_train_class=0,
):
    """Initialize directly as single prototypes (1 per class, rest masked)."""
    model.eval()
    emb_np = model(x, g).detach().cpu().numpy()
    model.train()

    if n_class > len(emb_np):
        n_class = len(emb_np)

    if proto_init == "class_kmeans":
        centers_np = _class_aware_centers(
            emb_np, labels_np, mask_lab_np, n_train_class,
            n_class, 1, seed)[:n_class]  # 1 prototype per class
    else:
        kmeans = KMeans(n_clusters=n_class, random_state=seed, n_init=10)
        kmeans.fit(emb_np)
        centers_np = kmeans.cluster_centers_
        print(f"  [Reinit-Single] KMeans inertia={kmeans.inertia_:.1f}")

    centers = torch.from_numpy(centers_np).float().to(device)
    centers_norm = F.normalize(centers, dim=-1)  # [n_class, D]

    for c in range(manager.n_class):
        if c < n_class:
            for v in range(n_views):
                idx = c * n_prototypes_max
                classifier.prototypes.data[v, idx] = centers_norm[c]
            manager.prototype_mask[c, 0] = 1
            manager.prototype_mask[c, 1:] = 0
            manager.alpha[c, 0] = 1.0
            manager.alpha[c, 1:] = 0.0
        else:
            manager.prototype_mask[c, :] = 0
            manager.alpha[c, :] = 0.0

    print(f"  [Reinit-Single] n_class={n_class}, mode={proto_init}")


# ==========================================================
# Progressive prototype merging: merge the most similar pair per class
#   merge_mode:
#     alpha      : alpha-weighted prototype-vector merge (default)
#     w_centroid : merged prototype = centroid of the union of the two
#                  prototypes' w-weighted node centroids
# ==========================================================
@torch.no_grad()
def merge_one_pair_per_class(manager, classifier, n_views,
                             merge_mode="alpha", num_w=None, den_w=None):
    proto = classifier.prototypes.data
    proto_view = proto.view(n_views, manager.n_class, manager.n_prototypes, manager.hidden_dim)

    # w_centroid mode: uses the chunk-accumulated w-weighted node centroid
    # statistics (num_w/den_w); no full w tensor required
    centroid = None
    if merge_mode == "w_centroid" and num_w is not None and den_w is not None:
        centroid = (num_w / (den_w.unsqueeze(-1) + 1e-12)).mean(dim=0)  # [C, K, D]

    for c in range(manager.n_class):
        active_idx = torch.where(manager.prototype_mask[c] > 0)[0]
        if len(active_idx) <= 1:
            continue

        # similarity from: view-averaged prototype vectors (alpha) or
        # node centroids (w_centroid)
        if centroid is not None:
            class_vec = centroid[c, active_idx]
        else:
            class_vec = proto_view[:, c, active_idx].mean(dim=0)
        class_vec = F.normalize(class_vec, dim=-1)
        sim_matrix = torch.matmul(class_vec, class_vec.t())
        sim_matrix.fill_diagonal_(-1)

        # most similar pair
        idx = sim_matrix.argmax()
        i = idx // sim_matrix.size(1)
        j = idx % sim_matrix.size(1)
        i_real = active_idx[i]
        j_real = active_idx[j]

        alpha_i = manager.alpha[c, i_real]
        alpha_j = manager.alpha[c, j_real]

        if merge_mode == "w_centroid" and num_w is not None:
            d_i = den_w[:, c, i_real].sum()
            d_j = den_w[:, c, j_real].sum()
            if d_i + d_j > 1e-6:
                merged = (num_w[:, c, i_real] + num_w[:, c, j_real]) / (d_i + d_j)
            else:
                # no node support, fall back to alpha-weighted vector merge
                merged = (alpha_i * proto_view[:, c, i_real]
                          + alpha_j * proto_view[:, c, j_real]) / (alpha_i + alpha_j + 1e-12)
        else:
            # alpha-weighted merge
            merged = (alpha_i * proto_view[:, c, i_real]
                      + alpha_j * proto_view[:, c, j_real]) / (alpha_i + alpha_j + 1e-12)

        proto_view[:, c, i_real] = merged
        manager.prototype_mask[c, j_real] = 0

    classifier.prototypes.data.copy_(
        proto_view.view(n_views, manager.n_class * manager.n_prototypes, manager.hidden_dim)
    )


# ==========================================================
# Graph structure metrics (dense ndarray and scipy CSR both supported)
# ==========================================================
def _to_csr(adj):
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    return adj


def graph_modularity(adj, labels):
    """Graph modularity Q = (sum_same A_ij - sum_c k_c^2/(2m)) / (2m)."""
    adj = _to_csr(adj)
    adj_no_self = adj.copy()
    adj_no_self.setdiag(0)
    adj_coo = adj_no_self.tocoo()

    m = adj_coo.data.sum() / 2.0
    if m < 1e-12:
        return 0.0
    k = np.asarray(adj_no_self.sum(axis=1)).flatten()

    same_mask = labels[adj_coo.row] == labels[adj_coo.col]
    sum_A = adj_coo.data[same_mask].sum()

    sum_k_sq = 0.0
    for c in np.unique(labels):
        k_c = k[labels == c].sum()
        sum_k_sq += k_c * k_c

    Q = (sum_A - sum_k_sq / (2.0 * m)) / (2.0 * m)
    return float(Q)


def graph_separation(adj, labels):
    """Graph separation S = sum_c (n_c/n) * (e_in_c / e_out_c)."""
    adj = _to_csr(adj)
    adj_no_self = adj.copy()
    adj_no_self.setdiag(0)
    adj_coo = adj_no_self.tocoo()

    n = len(labels)
    degree = np.asarray(adj_no_self.sum(axis=1)).flatten()
    unique_labels = np.unique(labels)

    label_to_idx = {c: i for i, c in enumerate(unique_labels)}
    e_in = np.zeros(len(unique_labels))
    same_mask = labels[adj_coo.row] == labels[adj_coo.col]
    for i in range(len(adj_coo.data)):
        if same_mask[i]:
            idx = label_to_idx[labels[adj_coo.row[i]]]
            e_in[idx] += adj_coo.data[i]

    S = 0.0
    for i, c in enumerate(unique_labels):
        nodes_c = (labels == c)
        n_c = nodes_c.sum()
        if n_c == 0:
            continue
        total_deg_c = degree[nodes_c].sum()
        e_out = total_deg_c - e_in[i]

        if e_out > 0:
            s_c = e_in[i] / e_out
        elif e_in[i] > 0:
            s_c = float(e_in[i])
        else:
            s_c = 0.0
        S += (n_c / n) * s_c
    return float(S)


# ==========================================================
# K estimation (Brent-accelerated candidate search)
# ==========================================================
def estimate_novel_class_count(
        emb, labels, mask_lab, n_train_class, adj, max_novel=10,
        score_weights=(1.0, 1.0, 1.0)
):
    w_mod, w_sep, w_acc = score_weights[:3]
    has_sc = len(score_weights) >= 4
    if has_sc:
        w_sc = score_weights[3]

    D_max = max(n_train_class + max_novel, int(labels.max()) + 1)
    cache = {}

    def _eval_k(K):
        K = int(K)
        if K < 1: K = 1
        if K > max_novel: K = max_novel
        if K in cache:
            return cache[K]

        n_clusters = n_train_class + K
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(emb)
        cluster_labels = kmeans.labels_

        mod = graph_modularity(adj, cluster_labels)
        sep = graph_separation(adj, cluster_labels)

        if n_clusters > 1 and n_clusters < len(emb):
            ch = metrics.calinski_harabasz_score(emb, cluster_labels)
        else:
            ch = 0.0

        if has_sc and n_clusters > 1 and n_clusters < len(emb):
            sc = metrics.silhouette_score(emb, cluster_labels)
        else:
            sc = -1.0

        y_pred_lab = cluster_labels[mask_lab]
        y_true_lab = labels[mask_lab]
        w_mat = np.zeros((D_max, D_max), dtype=int)
        for i in range(len(y_pred_lab)):
            w_mat[y_pred_lab[i], y_true_lab[i]] += 1
        row_ind, col_ind = linear_sum_assignment(w_mat.max() - w_mat)
        ind_map = {r: c for r, c in zip(row_ind, col_ind)}
        y_pred_aligned = np.array([ind_map.get(e, -1) for e in y_pred_lab])
        valid = y_pred_aligned >= 0
        acc = (y_pred_aligned[valid] == y_true_lab[valid]).mean() if valid.sum() > 0 else 0.0

        result = {
            'labels': cluster_labels,
            'mod': mod, 'sep': sep, 'ch': ch, 'sc': sc, 'acc': acc,
        }
        cache[K] = result
        return result

    def _proxy(K_float):
        K = int(round(K_float))
        K = max(1, min(K, max_novel))
        metrics_result = _eval_k(K)
        return -metrics_result['ch']

    try:
        result = minimize_scalar(
            _proxy, bounds=(1, max_novel), method='bounded',
            options={'xatol': 0.5, 'maxiter': 20}
        )
        K_brent = int(round(result.x))
        K_brent = max(1, min(K_brent, max_novel))
    except Exception:
        K_brent = max_novel // 2
        _eval_k(K_brent)

    for K in range(max(1, K_brent - 2), min(max_novel, K_brent + 2) + 1):
        _eval_k(K)

    Ks = sorted(cache.keys())
    mod_vals = np.array([cache[K]['mod'] for K in Ks])
    sep_vals = np.array([cache[K]['sep'] for K in Ks])
    acc_vals = np.array([cache[K]['acc'] for K in Ks])
    ch_vals = np.array([cache[K]['ch'] for K in Ks])

    def _norm(arr):
        rng = arr.max() - arr.min()
        if rng > 1e-12:
            return (arr - arr.min()) / rng
        return np.zeros_like(arr)

    mod_norm = _norm(mod_vals)
    sep_norm = _norm(sep_vals)
    acc_norm = _norm(acc_vals)

    if has_sc:
        sc_vals = np.array([cache[K]['sc'] for K in Ks])
        sc_norm = _norm(sc_vals)
        score = w_mod * mod_norm + w_sep * sep_norm + w_sc * sc_norm + w_acc * acc_norm
    else:
        score = w_mod * mod_norm + w_sep * sep_norm + w_acc * acc_norm

    best_idx = int(np.argmax(score))
    best_nnc = Ks[best_idx]
    best_n_pred = n_train_class + best_nnc

    scores_dict = {
        'candidate_nnc': Ks,
        'modularity': mod_vals.tolist(),
        'mod_norm': mod_norm.tolist(),
        'separation': sep_vals.tolist(),
        'sep_norm': sep_norm.tolist(),
        'labeled_acc': acc_vals.tolist(),
        'acc_norm': acc_norm.tolist(),
        'ch_score': ch_vals.tolist(),
        'joint_score': score.tolist(),
        'best_nnc': best_nnc,
        'best_n_pred': best_n_pred,
        'brent_K': K_brent,
        'n_evaluated': len(cache),
    }
    if has_sc:
        scores_dict['silhouette'] = sc_vals.tolist()
        scores_dict['sc_norm'] = sc_norm.tolist()

    return best_nnc, best_n_pred, scores_dict


# ==========================================================
# Train
# ==========================================================
def train(args):
    setup_seed(args.seed)

    @torch.no_grad()
    def hungarian_align_hard(pseudo_labels, true_labels, mask_lab, n_class):
        lab_idx = mask_lab.nonzero(as_tuple=True)[0]
        pred_lab = pseudo_labels[lab_idx].cpu().numpy()
        true_lab = true_labels[lab_idx].cpu().numpy()

        unique_pred = np.unique(pred_lab)
        unique_true = np.unique(true_lab)
        D_pred = int(max(unique_pred.max(), n_class - 1)) + 1
        D_true = int(max(unique_true.max(), n_class - 1)) + 1
        w = np.zeros((D_pred, D_true), dtype=np.int64)
        for i in range(len(pred_lab)):
            w[pred_lab[i], true_lab[i]] += 1

        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        map_dict = {row: col for row, col in zip(row_ind, col_ind)}

        aligned = torch.full_like(pseudo_labels, -1)
        for old_cls, new_cls in map_dict.items():
            aligned[pseudo_labels == old_cls] = new_cls

        mask_unmapped = (aligned == -1)
        aligned[mask_unmapped] = pseudo_labels[mask_unmapped]
        # clamp out-of-range labels to [0, n_class-1]
        aligned = torch.clamp(aligned, 0, n_class - 1)
        aligned[mask_lab] = true_labels[mask_lab]

        return aligned

    # GPU auto-detection
    if torch.cuda.is_available():
        try:
            device = torch.device(f"cuda:{args.device}")
            _ = torch.zeros(1).to(device)
            print(f"Using GPU: cuda:{args.device}")
        except Exception as e:
            device = torch.device("cpu")
            print(f"GPU check failed ({e}), falling back to CPU")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    (g, input_dim, n_class, n_train_class, mask_lab, mask_cls) = load_data(args.dataset, args.seed)

    args.input_dim = input_dim
    g = g.to(device)

    model = GNNModel(args).to(device)

    # N2P loss: single anchor (view 1), per-view averaging per variant
    criterion = N2PContrastiveLoss(
        tau=args.proto_tau,
        gamma=args.gamma,
        use_weight=args.use_weight,
        single_anchor=True,
        w_same_view=args.w_same_view,
        separate_view=args.variant_separate_view,
    ).to(device)

    # P2C loss: shared temperature, single anchor (view 1)
    p2c_criterion = P2CContrastiveLoss(
        tau=args.proto_tau,
        gamma=1.0,
        single_anchor=True,
        separate_view=args.variant_separate_view,
    ).to(device)

    # SupCon unsupervised pretraining loss (SimCLR-style)
    supcon_criterion = SupConLoss(
        device=device,
        temperature=args.supcon_tau,
        contrast_mode='all',
        base_temperature=args.supcon_tau
    ).to(device)

    # number of views
    if args.gat_view == 'head':
        n_views = args.num_gnn_heads
    elif args.gat_view == 'layer':
        n_views = args.num_gnn_layers
    else:
        n_views = 2

    # K estimation interval (warmup does not count into num_epochs)
    total_epochs = args.warmup_epochs + args.num_epochs
    fixed_k_mode = (not args.estimate_k)
    if args.estimate_k:
        k_refine_interval = max(1, args.num_epochs // args.k_refine_rounds)
        # short-run guard: when num_epochs < k_refine_rounds, do not refine
        # periodically (initial estimate only); otherwise a short run
        # triggers a full K estimation + prototype re-init every epoch
        if args.num_epochs < args.k_refine_rounds:
            k_refine_interval = args.num_epochs
        n_class_max = n_train_class + args.max_novel_k
        score_weights = tuple(float(w) for w in args.k_score_weights.split(','))

        # sparse symmetric adjacency (equivalent to the legacy dense version)
        N_nodes = g.num_nodes()
        src, dst = g.edges()
        adj = sp.csr_matrix(
            (np.ones(len(src), dtype=np.float64),
             (src.cpu().numpy(), dst.cpu().numpy())),
            shape=(N_nodes, N_nodes))
        adj = adj.maximum(adj.T)

        # multi-prototype phase rounds (B variants: first half multi, then single)
        if args.multi_proto_rounds < 0:
            multi_proto_rounds = (args.k_refine_rounds + 1) // 2
        else:
            multi_proto_rounds = min(args.multi_proto_rounds, args.k_refine_rounds)

        print(f"K estimation enabled: max classes={n_class_max}, "
              f"k_refine_rounds={args.k_refine_rounds}, k_refine_interval={k_refine_interval}, "
              f"merge_interval={args.merge_interval}, warmup={args.warmup_epochs}, "
              f"post-warmup epochs={args.num_epochs}, total={total_epochs}, "
              f"multi_proto_rounds={multi_proto_rounds}, "
              f"score_weights={score_weights}, proto_init={args.proto_init}, "
              f"mask_aware={args.mask_aware}, merge_mode={args.merge_mode}")
    else:
        k_refine_interval = args.num_epochs
        n_class_max = n_class
        score_weights = (1.0, 1.0, 1.0)
        # fixed-K: one multi-prototype init + progressive merging throughout
        multi_proto_rounds = 1 if args.merge_when_fixed_k else 0
        adj = None
        print(f"Fixed-K mode: n_class={n_class} (GT), "
              f"merge_when_fixed_k={args.merge_when_fixed_k}, "
              f"proto_init={args.proto_init}, mask_aware={args.mask_aware}")

    effective_n_class = n_train_class
    k_scores_dict = None
    k_history = []
    k_pred_round = 0
    k_changed_this_epoch = False

    classifier = ProtoClassifier(
        hidden_dim=args.hidden_dim,
        n_class=n_class_max,
        n_prototypes=args.n_prototypes,
        n_views=n_views
    ).to(device)

    # linear classification head (auxiliary CE supervision; output dim = GT)
    linear_cls = None
    if args.use_linear_cls and args.ce_weight > 0:
        linear_cls = torch.nn.Linear(args.hidden_dim, n_class).to(device)

    manager = PrototypeManager(
        n_class=n_class_max,
        n_prototypes=args.n_prototypes,
        hidden_dim=args.hidden_dim,
        tau=args.proto_tau,
        device=device
    ).to(device)

    x = g.ndata["feat"].float().to(device)
    labels = g.ndata["label"].to(device)
    labels_np = g.ndata["label"].cpu().numpy()
    mask_lab_tensor = torch.tensor(mask_lab).to(device)

    # prototype initialization
    print(f"Initializing prototypes (mode={args.proto_init}, "
          f"K={n_class_max} x M={args.n_prototypes})...")
    reinit_prototypes_multi(
        classifier, manager, model, g, x, labels_np, mask_lab,
        n_train_class, n_class_max, args.n_prototypes, n_views,
        args.hidden_dim, args.seed, device, proto_init=args.proto_init)
    manager.prototype_mask[n_class_max:] = 0

    opt_params = list(model.parameters()) + list(classifier.parameters())
    if linear_cls is not None:
        opt_params += list(linear_cls.parameters())
    optimizer = torch.optim.Adam(
        opt_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_all = 0
    best_old = 0
    best_new = 0
    best_geo = None

    csv_file = None
    if args.epochs_csv:
        csv_file = open(args.epochs_csv, "w")
        csv_file.write("epoch,all_acc,seen_acc,novel_acc,n_pred_class,"
                       "n2p_loss,p2c_loss,total_loss,seen_var,novel_var,IR,SR\n")

    global_class_proto = torch.zeros(n_views, n_class_max, args.hidden_dim, device=device)

    for epoch in range(total_epochs):
        model.train()
        classifier.train()
        if linear_cls is not None:
            linear_cls.train()

        # =====================================================
        # SupCon unsupervised pretraining warmup (SimCLR-style).
        # All nodes (seen + novel) participate equally in contrastive
        # learning, balancing intra-class variance.
        # =====================================================
        if args.supcon_warmup and epoch < args.warmup_epochs:
            features = model(x, g, view_type='head')  # [N, V, D]

            # large-graph sampling: the SupCon dot-product matrix is
            # [N*V, N*V] and can OOM on large graphs
            N_total = features.shape[0]
            max_n = args.warmup_max_nodes
            if 0 < max_n < N_total:
                rng = torch.Generator().manual_seed(args.seed + epoch)
                idx = torch.randperm(N_total, generator=rng)[:max_n].to(device)
                features_sample = features[idx]
            else:
                features_sample = features

            supcon_loss = supcon_criterion(features_sample)

            optimizer.zero_grad()
            supcon_loss.backward()
            optimizer.step()

            n_class_for_eval = effective_n_class if args.estimate_k else n_class
            all_acc, old_acc, new_acc, _geo = evaluate(
                model, g, labels, n_class_for_eval, mask_cls, device)

            if all_acc > best_all:
                best_all = all_acc
                best_old = old_acc
                best_new = new_acc

            if (epoch == 0 or (epoch + 1) % 5 == 0 or epoch == args.warmup_epochs - 1):
                print(f"[Warmup {epoch+1:02d}/{args.warmup_epochs}] SupCon | SupCon={supcon_loss.item():.4f} "
                      f"| All={all_acc:.4f} Seen={old_acc:.4f} Novel={new_acc:.4f}")

            continue  # skip the N2P/P2C path

        # =====================================================
        # Main training (main_epoch = 0 .. num_epochs-1)
        # =====================================================
        main_epoch = epoch - args.warmup_epochs

        # step 1: build multi-views
        views = build_views(
            model, g, x,
            args.edge_drop_rate, args.feat_mask_rate, device,
            gat_view=args.gat_view
        )

        # K estimation (iterative): first at main_epoch=0, then every
        # k_refine_interval epochs
        should_estimate = False
        if args.estimate_k:
            if main_epoch == 0:
                should_estimate = True
            elif main_epoch > 0 and main_epoch % k_refine_interval == 0:
                should_estimate = True

        k_changed_this_epoch = False
        do_fixed_k_init = fixed_k_mode and args.merge_when_fixed_k and main_epoch == 0

        if should_estimate:
            with torch.no_grad():
                model.eval()
                emb_np = model(x, g).detach().cpu().numpy()
                model.train()

            round_label = "Initial" if main_epoch == 0 else f"Refine @Epoch{main_epoch+1}"
            print(f"\n{'='*60}")
            print(f"[Epoch {main_epoch+1:03d}/{args.num_epochs}] {round_label} K estimation (Brent, via CH proxy)...")
            best_nnc, best_n_pred, k_scores_dict = estimate_novel_class_count(
                emb=emb_np, labels=labels_np, mask_lab=mask_lab,
                n_train_class=n_train_class, adj=adj,
                max_novel=args.max_novel_k, score_weights=score_weights
            )

            old_n_class = effective_n_class
            effective_n_class = best_n_pred
            k_pred_round += 1
            k_changed_this_epoch = (effective_n_class != old_n_class)

            manager.prototype_mask[effective_n_class:] = 0
            k_history.append((main_epoch, best_n_pred, best_nnc))

            print(f"{round_label} K estimation results (n_eval={k_scores_dict.get('n_evaluated', '?')}/{args.max_novel_k}):")
            print(f"  Evaluated novel counts: {k_scores_dict['candidate_nnc']}")
            print(f"  Modularity M(K)       : {[f'{x:.4f}' for x in k_scores_dict['modularity']]}")
            print(f"  Separation S(K)       : {[f'{x:.4f}' for x in k_scores_dict['separation']]}")
            print(f"  Labeled ACC (raw)     : {[f'{x:.4f}' for x in k_scores_dict['labeled_acc']]}")
            print(f"  Joint score I(K)      : {[f'{x:.4f}' for x in k_scores_dict['joint_score']]}")
            print(f"  >> Best novel classes = {best_nnc}, total n_pred = {effective_n_class}")
            if main_epoch > 0:
                print(f"  >> K changed: {old_n_class} → {effective_n_class}")
            print(f"  >> Ground truth: n_train_class={n_train_class}, n_class={n_class}")
            print(f"  >> K prediction round: {k_pred_round}/{args.k_refine_rounds}")
            if args.variant_multi_proto_mode == "always":
                phase_desc = "YES (always multi)"
            else:
                phase_desc = "YES" if k_pred_round <= multi_proto_rounds else "NO (single)"
            print(f"  >> multi_proto_phase={phase_desc}")
            print(f"{'='*60}\n")

            # phased prototype initialization
            use_multi_now = (args.variant_multi_proto_mode == "always"
                             or k_pred_round <= multi_proto_rounds)
            if use_multi_now:
                print(f"[Epoch {main_epoch+1:03d}/{args.num_epochs}] [Multi Phase] Re-initializing multi-prototypes...")
                reinit_prototypes_multi(
                    classifier, manager, model, g, x, labels_np, mask_lab,
                    n_train_class, effective_n_class, args.n_prototypes, n_views,
                    args.hidden_dim, args.seed + main_epoch, device,
                    proto_init=args.proto_init
                )
            else:
                print(f"[Epoch {main_epoch+1:03d}/{args.num_epochs}] [Single Phase] Re-initializing as single prototype per class...")
                reinit_prototypes_single(
                    classifier, manager, model, g, x, labels_np, mask_lab,
                    effective_n_class, args.n_prototypes, n_views,
                    args.hidden_dim, args.seed + main_epoch, device,
                    proto_init=args.proto_init, n_train_class=n_train_class
                )

            # reinit resets all masks, disable out-of-range classes again
            manager.prototype_mask[effective_n_class:] = 0
            active_num = manager.get_num_active()
            print(f"[Epoch {main_epoch+1:03d}/{args.num_epochs}] After reinit: "
                  f"active prototypes per class = {active_num.cpu().tolist()}, "
                  f"phase={'multi' if use_multi_now else 'single'}")

            # reset EMA when K changes
            global_class_proto = torch.zeros(n_views, n_class_max, args.hidden_dim, device=device)
        elif do_fixed_k_init:
            # fixed-K + merge_when_fixed_k: multi-prototype init, then
            # progressive merging for the rest of training
            k_pred_round = 1
            print(f"\n[Epoch {main_epoch+1:03d}/{args.num_epochs}] [Fixed-K] Multi-prototype init + progressive merge...")
            reinit_prototypes_multi(
                classifier, manager, model, g, x, labels_np, mask_lab,
                n_train_class, n_class, args.n_prototypes, n_views,
                args.hidden_dim, args.seed, device, proto_init=args.proto_init
            )
            global_class_proto = torch.zeros(n_views, n_class_max, args.hidden_dim, device=device)

        # =====================================================
        # steps 2-5: prototype assignment
        # =====================================================
        n_class_for_assign = n_class_max if args.estimate_k else n_class
        tau_q = args.proto_tau
        if args.q_gamma is not None:
            tau_q = args.proto_tau * (args.q_gamma ** main_epoch)  # A2: q temperature annealing
        (prototypes, proto_reshape, q, pseudo_labels, w_stats) = compute_prototype_assignment(
            classifier, manager, views,
            n_class_for_assign, args.n_prototypes, args.hidden_dim, n_train_class,
            tau=tau_q,
            only_seen=False,
            decouple_q_tau=args.decouple_q_tau
        )

        # alpha update (from chunked statistics, mathematically equivalent
        # to update_alpha)
        if args.hard_count:
            alpha_new = w_stats["hard_cnt"]
        else:
            alpha_new = w_stats["alpha_sum"]
        alpha_new = alpha_new * manager.prototype_mask[:n_class_for_assign]
        alpha_new = alpha_new / (alpha_new.sum(dim=-1, keepdim=True) + 1e-12)
        manager.alpha.copy_(alpha_new)

        # pseudo labels (main training: all nodes participate)
        n_class_for_eval = effective_n_class if args.estimate_k else n_class
        aligned_labels = hungarian_align_hard(
            pseudo_labels=pseudo_labels, true_labels=labels,
            mask_lab=mask_lab_tensor, n_class=n_class_for_eval
        )
        final_labels = labels.clone()
        final_labels[~mask_lab_tensor] = aligned_labels[~mask_lab_tensor]
        valid_mask = torch.ones_like(mask_lab_tensor, dtype=torch.bool)

        # =====================================================
        # class-level prototypes (w-weighted node centroids -> alpha aggregation)
        # num_w/den_w are chunk-accumulated by compute_prototype_assignment
        # =====================================================
        num_w = w_stats["num_w"]      # [V, C, K, D]
        den_w = w_stats["den_w"]      # [V, C, K]
        centroid_per_proto = num_w / (den_w.unsqueeze(-1) + 1e-12)
        C_eff_w = centroid_per_proto.size(1)
        alpha_sel = manager.alpha[:C_eff_w, :]
        alpha_exp = alpha_sel.unsqueeze(0).unsqueeze(-1)
        new_class_proto = (alpha_exp * centroid_per_proto).sum(dim=2)

        if C_eff_w < n_class_for_assign:
            pad = torch.zeros(n_views, n_class_for_assign - C_eff_w, args.hidden_dim, device=device)
            new_class_proto = torch.cat([new_class_proto, pad], dim=1)

        # =====================================================
        # EMA update (skipped at K-change boundaries)
        # =====================================================
        mu = args.ema_momentum

        with torch.no_grad():
            if main_epoch == 0:
                global_class_proto = new_class_proto
            else:
                if k_changed_this_epoch:
                    global_class_proto = new_class_proto
                    print(f"[Epoch {main_epoch+1:03d}/{args.num_epochs}] EMA RESET due to K change")
                else:
                    global_class_proto = mu * global_class_proto + (1 - mu) * new_class_proto

        # =====================================================
        # P2C loss (mask-aware: inactive prototypes are not anchors/negatives)
        # =====================================================
        p2c_mask = manager.prototype_mask if args.mask_aware else None
        if args.disable_p2c:
            p2c_loss = torch.tensor(0.0, device=device)
        else:
            p2c_loss = p2c_criterion(
                proto_reshape, global_class_proto, main_epoch,
                prototype_mask=p2c_mask)

        # =====================================================
        # N2P loss (mask-aware)
        # =====================================================
        if args.estimate_k:
            C_eff = effective_n_class
            proto_for_loss = prototypes[:, :C_eff * args.n_prototypes]
            proto_labels_for_loss = torch.arange(C_eff, device=device).repeat_interleave(args.n_prototypes)
        else:
            C_eff = n_class
            proto_for_loss = prototypes
            proto_labels_for_loss = classifier.prototype_labels.to(device)
        n2p_mask = manager.prototype_mask[:C_eff] if args.mask_aware else None

        n2p_loss = criterion(
            views=views[:, valid_mask],
            prototypes=proto_for_loss,
            prototype_labels=proto_labels_for_loss,
            labels=final_labels[valid_mask],
            epoch=main_epoch,
            prototype_mask=n2p_mask,
        )

        # =====================================================
        # Total loss: N2P + P2C (+ CE / diversity regularizers)
        # p2c_balance_mode: adaptive (small-graph B1) / fixed (arxiv B1)
        # =====================================================
        if args.disable_p2c:
            loss = n2p_loss
        elif args.p2c_balance_mode == "fixed":
            loss = n2p_loss + args.p2c_weight * p2c_loss
        else:
            balance = n2p_loss.detach() / (p2c_loss.detach() + 1e-8)
            loss = n2p_loss + args.p2c_weight * balance * p2c_loss

        # linear-head CE (seen ground-truth labels only, supervises encoder)
        cls_loss = torch.tensor(0.0, device=device)
        if linear_cls is not None and args.ce_weight > 0:
            emb_cls = views.mean(dim=0)  # [N, D]
            logits = linear_cls(emb_cls)  # [N, n_class]
            seen_cls_mask = mask_lab_tensor & valid_mask
            if seen_cls_mask.sum() > 0:
                cls_loss = F.cross_entropy(logits[seen_cls_mask], labels[seen_cls_mask])

        # prototype-guided CE (gradients flow directly into prototypes)
        proto_cls_loss = torch.tensor(0.0, device=device)
        if args.proto_ce_weight > 0:
            proto_centers = proto_reshape.mean(dim=0).mean(dim=1)  # [C, D] with grad
            emb_norm = F.normalize(views.mean(dim=0), dim=-1)
            centers_norm = F.normalize(proto_centers, dim=-1)
            proto_logits = emb_norm @ centers_norm.T / args.p2c_tau  # [N, C]
            seen_cls_mask = mask_lab_tensor & valid_mask
            if seen_cls_mask.sum() > 0:
                proto_cls_loss = F.cross_entropy(proto_logits[seen_cls_mask], labels[seen_cls_mask])

        # EMA class-center guided CE (gradients flow into the encoder; off by default)
        center_cls_loss = torch.tensor(0.0, device=device)
        if args.center_ce_weight > 0:
            centers_ce = global_class_proto.mean(dim=0).detach()  # [C, D] no grad
            center_logits = F.normalize(views.mean(dim=0), dim=-1) @ F.normalize(centers_ce, dim=-1).T / args.p2c_tau
            seen_ce_mask = mask_lab_tensor & valid_mask
            if seen_ce_mask.sum() > 0:
                center_cls_loss = F.cross_entropy(center_logits[seen_ce_mask], labels[seen_ce_mask])

        # intra-class prototype diversity (penalizes similarity within a class)
        proto_div_loss = torch.tensor(0.0, device=device)
        if args.proto_div_weight > 0:
            proto_norm = F.normalize(proto_reshape, dim=-1)  # [V, C, K, D]
            proto_sim = torch.einsum('vckd,vcmd->vckm', proto_norm, proto_norm)
            mask_pd = 1 - torch.eye(args.n_prototypes, device=device).unsqueeze(0).unsqueeze(0)
            pos_sim_pd = torch.relu(proto_sim * mask_pd)
            proto_div_loss = pos_sim_pd.mean() + pos_sim_pd.pow(2).mean()

        # inter-class center diversity (penalizes similarity between classes)
        center_div_loss = torch.tensor(0.0, device=device)
        if args.center_div_weight > 0:
            centers_cd = F.normalize(global_class_proto.mean(dim=0), dim=-1)  # [C, D]
            center_sim = centers_cd @ centers_cd.T
            mask_cd = 1 - torch.eye(centers_cd.shape[0], device=device)
            pos_sim_cd = torch.relu(center_sim[mask_cd.bool()])
            center_div_loss = pos_sim_cd.mean() + pos_sim_cd.pow(2).mean()

        loss = (loss + args.ce_weight * cls_loss
                + args.proto_ce_weight * proto_cls_loss
                + args.center_ce_weight * center_cls_loss
                + args.proto_div_weight * proto_div_loss
                + args.center_div_weight * center_div_loss)

        if args.entropy_weight > 0:
            # L_entropy = -H(p) = sum(p * log p): p*log(p) < 0, so minimizing
            # L_entropy flattens the marginal class distribution p (anti-collapse)
            p_marginal = q.mean(dim=0)  # [C] average class distribution over nodes
            marginal_entropy = (p_marginal * torch.log(p_marginal + 1e-12)).sum()
            loss = loss + args.entropy_weight * marginal_entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # =====================================================
        # Progressive prototype merging (multi-prototype phase, one pair
        # per class every merge_interval epochs)
        # =====================================================
        if args.variant_multi_proto_mode == "always":
            in_multi_phase = (k_pred_round > 0)
        else:
            in_multi_phase = (k_pred_round > 0 and k_pred_round <= multi_proto_rounds)
        if (in_multi_phase
                and main_epoch > (k_pred_round - 1) * k_refine_interval
                and (main_epoch - (k_pred_round - 1) * k_refine_interval) % args.merge_interval == 0):
            active_per_class = manager.prototype_mask.sum(dim=1)
            if active_per_class.max() > 1:
                merge_one_pair_per_class(
                    manager, classifier, n_views,
                    merge_mode=args.merge_mode,
                    num_w=w_stats["num_w"], den_w=w_stats["den_w"]
                )
                active_after = manager.prototype_mask.sum(dim=1)
                n_merged = (active_per_class - active_after).sum().item()
                print(f"[Epoch {main_epoch+1:03d}/{args.num_epochs}] Progressive Merge ({args.merge_mode}): "
                      f"{n_merged} protos merged, "
                      f"active per class = {active_after.cpu().tolist()}")

        # =====================================================
        # Evaluation (+ geometry metrics IR/SR)
        # frequency: every eval_freq epochs + last epoch
        # =====================================================
        do_eval = (main_epoch % args.eval_freq == 0
                   or main_epoch == args.num_epochs - 1)
        if do_eval:
            proto_init_eval = global_class_proto if args.eval_proto_init else None
            all_acc, old_acc, new_acc, geo = evaluate(model, g, labels, n_class_for_eval, mask_cls, device,
                                                     proto_init=proto_init_eval)

            if all_acc > best_all:
                best_all = all_acc
                best_old = old_acc
                best_new = new_acc
                best_geo = geo

        # Print: full info every 5 epochs or on special events; loss summary otherwise
        active_num = manager.get_num_active()
        phase_str = "multi" if in_multi_phase else "single"
        k_status = f"K_changed={k_changed_this_epoch}"
        q_entropy = -(q * torch.log(q + 1e-12)).sum(dim=1).mean()
        n_pred_reported = effective_n_class if args.estimate_k else n_class
        if do_eval and (main_epoch == 0 or (main_epoch + 1) % 5 == 0
                        or main_epoch == args.num_epochs - 1 or should_estimate):
            print(f"[Epoch {main_epoch+1:03d}/{args.num_epochs}] {k_status} [{phase_str}] | N2P={n2p_loss.item():.4f} P2C={p2c_loss.item():.4f} "
                  f"CLS={cls_loss.item():.4f} PCLS={proto_cls_loss.item():.4f} CCLS={center_cls_loss.item():.4f} "
                  f"PDIV={proto_div_loss.item():.4f} DIV={center_div_loss.item():.4f} "
                  f"Loss={loss.item():.4f} | All={all_acc:.4f} Seen={old_acc:.4f} Novel={new_acc:.4f} "
                  f"n_class={n_pred_reported} | Entropy={q_entropy:.4f} | ActiveProto sum={int(active_num.sum())}")
            print(f"            Geometry: seen_var={geo['seen_var']:.4f} novel_var={geo['novel_var']:.4f} "
                  f"IR={geo['IR']:.3f} SR={geo['SR']:.3f}")
        elif not do_eval and ((main_epoch + 1) % 5 == 0 or should_estimate):
            print(f"[Epoch {main_epoch+1:03d}/{args.num_epochs}] {k_status} [{phase_str}] | N2P={n2p_loss.item():.4f} P2C={p2c_loss.item():.4f} "
                  f"CLS={cls_loss.item():.4f} PCLS={proto_cls_loss.item():.4f} DIV={center_div_loss.item():.4f} "
                  f"Loss={loss.item():.4f} | n_class={n_pred_reported} | Entropy={q_entropy:.4f}")

        # CSV: write on evaluation epochs only
        if csv_file is not None and do_eval:
            n_pred_reported = effective_n_class if args.estimate_k else n_class
            csv_file.write(f"{main_epoch},{all_acc:.6f},{old_acc:.6f},{new_acc:.6f},"
                           f"{n_pred_reported},{n2p_loss.item():.6f},"
                           f"{p2c_loss.item():.6f},{loss.item():.6f},"
                           f"{geo['seen_var']:.6f},{geo['novel_var']:.6f},"
                           f"{geo['IR']:.6f},{geo['SR']:.6f}\n")
            csv_file.flush()

    if csv_file is not None:
        csv_file.close()

    return best_all, best_old, best_new, effective_n_class, n_train_class, n_class, best_geo


# ==========================================================
# Main (multi-seed)
# ==========================================================
def main(forced_dataset=None, forced_variant=None):
    args = parse_args(forced_dataset, forced_variant)
    if args.single_run:
        all_seed = [args.seed]
    else:
        all_seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    all_results = []
    k_estimates = []
    os.makedirs("results", exist_ok=True)
    tag = f"{args.variant}"
    f = open(f"results/results_{args.dataset}_{tag}.txt", "a+")

    for seed in all_seed:
        print("=" * 60)
        print(f"Running seed {seed}")
        args.seed = seed
        results = train(args)
        best_all, best_old, best_new, n_pred, n_train_class_gt, n_class_gt, best_geo = results
        all_results.append([best_all, best_old, best_new])
        k_estimates.append(n_pred)
        print(f"Seed {seed} Result: All={best_all:.4f}, Seen={best_old:.4f}, Novel={best_new:.4f}")
        f.write(f"seed: {seed}\toverall_acc:{best_all:.4f}seen_acc:{best_old:.4f}unseen_acc:{best_new:.4f}n_pred_class:{n_pred}\n")
        f.flush()
        if args.estimate_k:
            print(f"Seed {seed} Estimated total classes = {n_pred} "
                  f"(novel={n_pred - n_train_class_gt}, gt_total={n_class_gt})")
        if best_geo is not None:
            print(f"Seed {seed} Best-epoch geometry: IR={best_geo['IR']:.3f} SR={best_geo['SR']:.3f} "
                  f"seen_var={best_geo['seen_var']:.4f} novel_var={best_geo['novel_var']:.4f}")

    all_results = np.array(all_results)
    mean_results = np.mean(all_results, axis=0)
    std_results = np.std(all_results, axis=0)

    print("\n" + "=" * 60)
    print(f"Final Results:\n"
          f"Overall ACC: {mean_results[0]:.4f} ± {std_results[0]:.4f}\n"
          f"Seen ACC:    {mean_results[1]:.4f} ± {std_results[1]:.4f}\n"
          f"Novel ACC:   {mean_results[2]:.4f} ± {std_results[2]:.4f}")
    k_arr = np.array(k_estimates, dtype=float) if k_estimates else np.array([])
    if args.estimate_k:
        print(f"\nEstimated total classes (n_pred = n_seen + n_novel):")
        print(f"  Mean ± Std: {k_arr.mean():.2f} ± {k_arr.std():.2f}")
        print(f"  Per-seed values: {k_arr.tolist()}")
        print(f"  Ground truth:   {n_class_gt} (n_train_class={n_train_class_gt})")

    f.write(f"Overall: "
            f"overall_acc:{mean_results[0]:.4f}+/-{std_results[0]:.4f}"
            f"seen_acc:{mean_results[1]:.4f}+/-{std_results[1]:.4f}"
            f"unseen_acc:{mean_results[2]:.4f}+/-{std_results[2]:.4f}"
            f"n_pred_class:{k_arr.mean():.4f}+/-{k_arr.std():.4f}\n")
    f.close()


if __name__ == "__main__":
    main()
