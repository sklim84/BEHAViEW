import argparse

parser = argparse.ArgumentParser()

# Commons
parser.add_argument('--model_name', type=str, default='GRACE_w_cen')
parser.add_argument('--node_data_name', type=str, default='HOFINET_NODE_FEAT')
parser.add_argument('--edge_data_name', type=str, default='HOFINET_EDGES')
parser.add_argument('--metric_save_path', type=str, default='./results/exp_results.csv')
parser.add_argument('--loss', type=str, default='InfoNCE')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--struct_feats', nargs='+',
                    help="List of structural feature types (dc, pagerank, hits_hub, hits_auth, kcore, triangle, betweenness)",
                    default=['dc', 'pagerank', 'hits_hub', 'hits_auth', 'kcore', 'triangle', 'betweenness'])

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--input_dim', type=int, default=16)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--gconv_nlayers', type=int, default=3)

# GRACE
parser.add_argument('--proj_dim', type=int, default=32)

# k-NN graph view
parser.add_argument('--knn_graph', type=str, default=None,
                    help='k-NN graph file name without .csv (e.g., HOFINET_KNN_BEHAV_k10)')
parser.add_argument('--encoder_type', type=str, default='bgrl',
                    choices=['bgrl', 'dgi', 'mvgrl', 'gbt', 'grace', 'dgi_bn', 'mvgrl_bn', 'grace_bn', 'gca', 'gin'],
                    help='GNN encoder type for subgraph_cl (bgrl/dgi/mvgrl)')
parser.add_argument('--subgraph_pool', action='store_true',
                    help='Use subgraph (neighborhood) pooling instead of global pooling')

# Evaluation
parser.add_argument('--skip_tsne', action='store_true', help='Skip t-SNE visualization (for HP search)')


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
