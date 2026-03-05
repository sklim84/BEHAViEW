seed=2025
gpu='0'

metric_save_path='./_results/exp_results_rq5.csv'

node_data_name='HF_FND_2024_03_SAMPLE_100K_NODE_FEAT'
edge_data_name='HF_FND_2024_03_SAMPLE_100K'

# BGRL
python -u BGRL_L2L_w_cen.py \
              --model_name 'BGRL_L2L_w_cen' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name \
              --cen_feats "dc cc bc" \
              --lr 0.1 \
              --input_dim 16 \
              --hidden_dim 512 \
              --gconv_nlayers 1 \
              --loss BarlowTwins

python -u BGRL_L2L_w_org.py \
              --model_name 'BGRL_L2L_w_org' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name

# DGI-IND
python -u DGI_inductive_w_cen.py \
              --model_name 'DGI_inductive_w_cen' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name \
              --cen_feats "dc" \
              --lr 0.01 \
              --input_dim 8 \
              --hidden_dim 128 \
              --gconv_nlayers 5 \
              --loss JSD

python -u DGI_inductive_w_org.py \
              --model_name 'DGI_inductive_w_org' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name

# DGI-TRS
python -u DGI_transductive_w_cen.py \
              --model_name 'DGI_transductive_w_cen' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name \
              --cen_feats "dc cc bc" \
              --lr 0.001 \
              --input_dim 4 \
              --hidden_dim 256 \
              --gconv_nlayers 3 \
              --loss JSD

python -u DGI_transductive_w_org.py \
              --model_name 'DGI_transductive_w_org' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name

# GBT
python -u GBT_w_cen.py \
              --model_name 'GBT_w_cen' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name \
              --cen_feats "dc cc bc" \
              --lr 5e-4 \
              --input_dim 16 \
              --hidden_dim 128 \
              --loss BarlowTwins

python -u GBT_w_org.py \
              --model_name 'GBT_w_org' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \

# GRACE
python -u GRACE_w_cen.py \
                --model_name 'GRACE_w_cen' \
                --seed $seed \
                --gpu $gpu \
                --metric_save_path $metric_save_path \
                --node_data_name $node_data_name \
                --edge_data_name $edge_data_name \
                --cen_feats "cc" \
                --lr 0.0001 \
                --input_dim 4 \
                --hidden_dim 64 \
                --gconv_nlayers 2 \
                --proj_dim 32 \
                --loss InfoNCE

python -u GRACE_w_org.py \
                --model_name 'GRACE_w_org' \
                --seed $seed \
                --gpu $gpu \
                --metric_save_path $metric_save_path \
                --node_data_name $node_data_name

# MVGRL
python -u MVGRL_node_w_cen.py \
              --model_name 'MVGRL_node_w_cen' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name \
              --cen_feats "dc cc bc" \
              --lr 0.0001 \
              --hidden_dim 128 \
              --gconv_nlayers 4 \
              --loss BootstrapLatent

python -u MVGRL_node_w_org.py \
              --model_name 'MVGRL_node_w_org' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name

