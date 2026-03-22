seed=2025
gpu='0'

metric_save_path='./results/exp_results_rq5.csv'

node_data_name='HOFINET_NODE_FEAT'
edge_data_name='HOFINET_EDGES'

struct_feats="dc cc pagerank hits_hub hits_auth kcore triangle"

# BGRL
python -u models/bgrl_w_cen.py \
              --model_name 'bgrl_w_cen' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name \
              --struct_feats $struct_feats \
              --lr 0.1 \
              --input_dim 16 \
              --hidden_dim 512 \
              --gconv_nlayers 1 \
              --loss BarlowTwins

python -u models/bgrl_w_org.py \
              --model_name 'bgrl_w_org' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name

# DGI-IND
python -u models/dgi_inductive_w_cen.py \
              --model_name 'dgi_inductive_w_cen' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name \
              --struct_feats $struct_feats \
              --lr 0.01 \
              --input_dim 8 \
              --hidden_dim 128 \
              --gconv_nlayers 5 \
              --loss JSD

python -u models/dgi_inductive_w_org.py \
              --model_name 'dgi_inductive_w_org' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name

# DGI-TRS
python -u models/dgi_transductive_w_cen.py \
              --model_name 'dgi_transductive_w_cen' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name \
              --struct_feats $struct_feats \
              --lr 0.001 \
              --input_dim 4 \
              --hidden_dim 256 \
              --gconv_nlayers 3 \
              --loss JSD

python -u models/dgi_transductive_w_org.py \
              --model_name 'dgi_transductive_w_org' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name

# GBT
python -u models/gbt_w_cen.py \
              --model_name 'gbt_w_cen' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name \
              --struct_feats $struct_feats \
              --lr 5e-4 \
              --input_dim 16 \
              --hidden_dim 128 \
              --loss BarlowTwins

python -u models/gbt_w_org.py \
              --model_name 'gbt_w_org' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name \

# GRACE
python -u models/grace_w_cen.py \
                --model_name 'grace_w_cen' \
                --seed $seed \
                --gpu $gpu \
                --metric_save_path $metric_save_path \
                --node_data_name $node_data_name \
                --edge_data_name $edge_data_name \
                --struct_feats $struct_feats \
                --lr 0.0001 \
                --input_dim 4 \
                --hidden_dim 64 \
                --gconv_nlayers 2 \
                --proj_dim 32 \
                --loss InfoNCE

python -u models/grace_w_org.py \
                --model_name 'grace_w_org' \
                --seed $seed \
                --gpu $gpu \
                --metric_save_path $metric_save_path \
                --node_data_name $node_data_name \
                --edge_data_name $edge_data_name

# MVGRL
python -u models/mvgrl_w_cen.py \
              --model_name 'mvgrl_w_cen' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name \
              --struct_feats $struct_feats \
              --lr 0.0001 \
              --hidden_dim 128 \
              --gconv_nlayers 4 \
              --loss BootstrapLatent

python -u models/mvgrl_w_org.py \
              --model_name 'mvgrl_w_org' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name
