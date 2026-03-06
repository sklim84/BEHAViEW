import os
import pandas as pd
from datetime import datetime
import random
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import GCL.losses as L


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_loss(loss_name):
    """손실 함수 이름으로부터 GCL Loss 객체를 생성한다."""
    if loss_name == 'BootstrapLatent':
        return L.BootstrapLatent()
    elif loss_name == 'JSD':
        return L.JSD()
    elif loss_name == 'BarlowTwins':
        return L.BarlowTwins()
    elif loss_name == 'InfoNCE':
        return L.InfoNCE(tau=0.2)
    raise ValueError(f"Unknown loss: {loss_name}")


def build_result_dict(model_name, args, test_result, ari_score, sil_score, use_cen=True):
    """실험 결과 딕셔너리를 구성한다."""
    return {
        'Model': model_name,
        'Data': args.node_data_name,
        'Seed': args.seed,
        'cen_feats': args.cen_feats if use_cen else "None",
        'lr': args.lr if use_cen else -1,
        'input_dim': getattr(args, 'input_dim', -1) if use_cen else -1,
        'hidden_dim': args.hidden_dim if use_cen else -1,
        'proj_dim': getattr(args, 'proj_dim', -1),
        'gconv_nlayers': args.gconv_nlayers if use_cen else -1,
        'loss': args.loss,
        'pre_1': test_result['pre_1'],
        'rec_1': test_result['rec_1'],
        'f1_1': test_result['f1_1'],
        'pre_0': test_result['pre_0'],
        'rec_0': test_result['rec_0'],
        'f1_0': test_result['f1_0'],
        'F1Mi': test_result['micro_f1'],
        'F1Ma': test_result['macro_f1'],
        'auroc': test_result['auroc'],
        'auprc': test_result['auprc'],
        'ari_score': ari_score,
        'sil_score': sil_score
    }


def save_results_to_csv(results, file_path):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for result in results:
        result_with_time = {'timestamp': current_time}
        result_with_time.update(result)
        result.clear()
        result.update(result_with_time)

    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_new = pd.DataFrame(results)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = pd.DataFrame(results)

    df_combined.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")


def evaluate_with_metrics(z, y, split):
    """
    z: 노드 임베딩 (torch.Tensor)
    y: 노드 레이블 (torch.Tensor)
    split: dict with keys 'train' and 'test' (torch.Tensor of indices)
    """
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    train_idx = split['train']
    test_idx = split['test']

    if isinstance(train_idx, torch.Tensor):
        train_idx = train_idx.cpu().numpy()
    if isinstance(test_idx, torch.Tensor):
        test_idx = test_idx.cpu().numpy()

    z_train, y_train = z[train_idx], y[train_idx]
    z_test, y_test = z[test_idx], y[test_idx]

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(z_train, y_train)

    y_pred = clf.predict(z_test)
    y_score = clf.predict_proba(z_test)[:, 1]

    print("\n Classification Report")
    print(classification_report(y_test, y_pred, digits=4))
    clf_report = classification_report(y_test, y_pred, digits=4, output_dict=True)

    f1_micro = f1_score(y_test, y_pred, average="micro")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_by_class = f1_score(y_test, y_pred, average=None)

    print(f"F1-micro: {f1_micro:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")
    print(f"F1-by-class [Normal, Fraud]: {f1_by_class}")

    print("\n AUROC & AUPRC")
    try:
        auc = roc_auc_score(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        print(f"AUROC: {auc:.4f}")
        print(f"AUPRC: {ap:.4f}")
    except Exception as e:
        print("AUROC/AUPRC 계산 실패:", e)

    print("\n Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    return {
        'pre_0': clf_report['0']['precision'],
        'rec_0': clf_report['0']['recall'],
        'f1_0': clf_report['0']['f1-score'],
        'pre_1': clf_report['1']['precision'],
        'rec_1': clf_report['1']['recall'],
        'f1_1': clf_report['1']['f1-score'],
        "micro_f1": f1_micro,
        "macro_f1": f1_macro,
        "auroc": auc if 'auc' in locals() else None,
        "auprc": ap if 'ap' in locals() else None
    }


def visualize_tsne(seed, embeddings, labels, save_path=None, max_samples=50000, skip=False):
    if skip:
        return 0.0, 0.0
    y_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    if len(embeddings) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(embeddings), max_samples, replace=False)
        emb_sub = embeddings[idx]
        y_sub = y_np[idx]
        print(f"[TSNE] Subsampled {max_samples} from {len(embeddings)} nodes")
    else:
        emb_sub = embeddings
        y_sub = y_np

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=seed)
    z_2d = tsne.fit_transform(emb_sub)

    kmeans = KMeans(n_clusters=2, random_state=seed).fit(z_2d)
    ari_score = adjusted_rand_score(y_sub, kmeans.labels_)
    sil_score = silhouette_score(z_2d, y_sub)

    print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
    print(f"Silhouette Score: {sil_score:.4f}")

    plt.figure(figsize=(4, 4))
    plt.scatter(z_2d[y_sub == 0, 0], z_2d[y_sub == 0, 1], c='#2166ac', label='Benign', alpha=1, s=5)
    plt.scatter(z_2d[y_sub == 1, 0], z_2d[y_sub == 1, 1], c='#ef4136', label='Fraud', alpha=1, s=5)
    plt.axis('off')
    plt.tight_layout()
    plt.margins(0)
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return ari_score, sil_score
