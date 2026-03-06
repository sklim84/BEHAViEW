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
parser.add_argument('--cen_feats', nargs='+',
                    help="List of graph feature types (dc, cc, pagerank, hits_hub, hits_auth, kcore, triangle)",
                    default=['dc', 'cc', 'pagerank', 'hits_hub', 'hits_auth', 'kcore', 'triangle'])

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--input_dim', type=int, default=16)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--gconv_nlayers', type=int, default=3)

# GRACE
parser.add_argument('--proj_dim', type=int, default=32)


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
