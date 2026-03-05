import argparse

parser = argparse.ArgumentParser()

# Commons
parser.add_argument('--model_name', type=str, default='GRACE_w_cen_me')
# HF_FND_2024_03_SAMPLE_100K_NODE_FEAT
parser.add_argument('--node_data_name', type=str, default='HF_FND_2024_03_SP_H1_NODE_FEAT')
parser.add_argument('--edge_data_name', type=str, default='HF_FND_2024_03_SP_H1')
parser.add_argument('--metric_save_path', type=str, default='./_results/exp_results_GRACE.csv')
# BootstrapLatent, JSD, BarlowTwins, InfoNCE
parser.add_argument('--loss', type=str, default='InfoNCE')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--cen_feats', nargs='+', help="List of centrality types (dc, cc, bc)",
                    default=['dc', 'cc', 'bc']) # default=['dc', 'cc', 'bc']

# parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--input_dim', type=int, default=16)

parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--gconv_nlayers', type=int, default=3) # 3

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
