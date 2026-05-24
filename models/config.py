import argparse

parser = argparse.ArgumentParser()

# Commons
parser.add_argument('--model_name', type=str, default='GRACE_w_cen')
parser.add_argument('--node_data_name', type=str, default='amlworld/AMLWORLD_NODE_FEAT')
parser.add_argument('--edge_data_name', type=str, default='amlworld/AMLWORLD_EDGES')
parser.add_argument('--metric_save_path', type=str, default='./results/exp_results.csv')
parser.add_argument('--loss', type=str, default='BootstrapLatent',
                    choices=['BootstrapLatent', 'BarlowTwins', 'InfoNCE', 'JSD'])
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--device', type=str, default='auto',
                    choices=['auto', 'cuda', 'mps', 'cpu'],
                    help='Training device. auto prefers CUDA, then MPS, then CPU.')
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--struct_feats', nargs='+',
                    help="List of structural feature types (dc, pagerank, hits_hub, hits_auth, kcore, triangle, betweenness)",
                    default=['dc', 'pagerank', 'hits_hub', 'hits_auth', 'kcore', 'triangle', 'betweenness'])

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--input_dim', type=int, default=16)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--gconv_nlayers', type=int, default=3)

# GRACE
parser.add_argument('--proj_dim', type=int, default=32)

# k-NN graph view
parser.add_argument('--knn_graph', type=str, default=None,
                    help='k-NN graph file name without .csv (e.g., amlworld/AMLWORLD_KNN_BEHAV_k10)')
parser.add_argument('--encoder_type', type=str, default='bgrl',
                    choices=['bgrl', 'dgi', 'mvgrl', 'gbt', 'grace', 'dgi_bn', 'mvgrl_bn', 'grace_bn', 'gca', 'gin'],
                    help='GNN encoder type for subgraph_cl (bgrl/dgi/mvgrl)')
parser.add_argument('--subgraph_pool', action='store_true',
                    help='Use subgraph (neighborhood) pooling instead of global pooling')
parser.add_argument('--pool_variant', type=str, default='mean',
                    choices=['mean', 'heterophily', 'cycle'],
                    help='Subgraph pool variant: mean (default), heterophily-aware (cosine-sim weighted), or cycle-aware (HAP + triangle boost)')
parser.add_argument('--cycle_alpha', type=float, default=2.0,
                    help='Triangle-membership boost coefficient for cycle-aware pool')

# Evaluation
parser.add_argument('--tune_threshold', action='store_true',
                    help='Tune LogisticRegression decision threshold on validation split for suspicious-class F1')
parser.add_argument('--skip_tsne', action='store_true', help='Skip t-SNE visualization (for HP search)')
parser.add_argument('--save_embeddings_to', type=str, default=None,
                    help='If set, save the final node embeddings z (n, 2*hidden) as .npz to this path.')
parser.add_argument('--train_ratio', type=float, default=0.1, help='Train split ratio (val=same, test=1-2*train)')
parser.add_argument('--split_path', type=str, default=None,
                    help='Optional .npz file with train/valid/test node indices, e.g. temporal split.')


def get_config():
    return parser.parse_args()


def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp
