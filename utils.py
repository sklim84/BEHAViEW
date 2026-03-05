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

# 시드 고정 함수
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results_to_csv(results, file_path):
    # 현재 시간 추가
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 각 결과에 현재 시간을 첫 번째 컬럼으로 추가
    for result in results:
        # timestamp를 가장 먼저 추가한 새로운 딕셔너리 생성
        result_with_time = {'timestamp': current_time}
        result_with_time.update(result)  # 기존 결과 데이터를 뒤에 추가
        result.clear()
        result.update(result_with_time)  # 업데이트된 결과로 덮어씀

    # 파일이 이미 존재하는지 확인
    if os.path.exists(file_path):
        # 기존 파일을 불러오기
        df_existing = pd.read_csv(file_path)
        # 새로운 결과와 기존 파일을 병합
        df_new = pd.DataFrame(results)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        # 파일이 없으면 새로운 데이터프레임 생성
        df_combined = pd.DataFrame(results)

    # 결과를 CSV 파일로 저장
    df_combined.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")



def evaluate_with_metrics(z, y, split):
    """
    z: 노드 임베딩 (torch.Tensor)
    y: 노드 레이블 (torch.Tensor)
    split: dict with keys 'train' and 'test' (torch.Tensor of indices)
    """

    # Tensor → numpy 변환
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    # split dict로부터 인덱스 추출
    train_idx = split['train']
    test_idx = split['test']

    # torch.Tensor → numpy 정수 배열
    if isinstance(train_idx, torch.Tensor):
        train_idx = train_idx.cpu().numpy()
    if isinstance(test_idx, torch.Tensor):
        test_idx = test_idx.cpu().numpy()

    # 인덱싱
    z_train, y_train = z[train_idx], y[train_idx]
    z_test, y_test = z[test_idx], y[test_idx]

    # Logistic Regression with class_weight='balanced' to address fraud imbalance
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(z_train, y_train)

    y_pred = clf.predict(z_test)
    y_score = clf.predict_proba(z_test)[:, 1]  # fraud 확률

    print("\n📊 Classification Report")
    print(classification_report(y_test, y_pred, digits=4))
    clf_report = classification_report(y_test, y_pred, digits=4, output_dict=True)

    f1_micro = f1_score(y_test, y_pred, average="micro")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_by_class = f1_score(y_test, y_pred, average=None)

    print(f"F1-micro: {f1_micro:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")
    print(f"F1-by-class [Normal, Fraud]: {f1_by_class}")

    print("\n🧪 AUROC & AUPRC")
    try:
        auc = roc_auc_score(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        print(f"AUROC: {auc:.4f}")
        print(f"AUPRC: {ap:.4f}")
    except Exception as e:
        print("AUROC/AUPRC 계산 실패:", e)

    print("\n🧾 Confusion Matrix")
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

def visualize_tsne(seed, embeddings: np.ndarray, labels: torch.Tensor, save_path=None):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=seed)
    z_2d = tsne.fit_transform(embeddings)
    y_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    # ARI
    kmeans = KMeans(n_clusters=2, random_state=seed).fit(z_2d)
    ari_score = adjusted_rand_score(y_np, kmeans.labels_)
    sil_score = silhouette_score(z_2d, y_np)
    # sil_score = silhouette_score(z_2d, kmeans.labels_)

    print(f"🔍 Adjusted Rand Index (ARI): {ari_score:.4f}")
    print(f"📐 Silhouette Score: {sil_score:.4f}")

    plt.figure(figsize=(4, 4))
    plt.scatter(z_2d[y_np == 0, 0], z_2d[y_np == 0, 1], c='#2166ac', label='Benign', alpha=1, s=5)
    plt.scatter(z_2d[y_np == 1, 0], z_2d[y_np == 1, 1], c='#ef4136', label='Fraud', alpha=1, s=5)
    # plt.legend()
    # plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.margins(0)
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return ari_score, sil_score