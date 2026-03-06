seed=2025
gpu='0'

metric_save_path='./results/exp_results_GRACE.csv'

#node_data_name='HF_FND_2024_03_SAMPLE_100K_NODE_FEAT'
#edge_data_name='HF_FND_2024_03_SAMPLE_100K'
node_data_name='HF_FND_2024_03_SP_H1_NODE_FEAT'
edge_data_name='HF_FND_2024_03_SP_H1'

# "dc" "cc" "bc" "dc cc" "dc bc" "cc bc" "dc cc bc"
for cen_feats in "dc" "cc" "bc" "dc cc" "dc bc" "cc bc" "dc cc bc"; do
  for lr in 0.01 0.001; do # 0.01
    for input_dim in 4 8 16; do
      for hidden_dim in 32 64 128; do # 32
        for gconv_nlayers in 2 4 8; do # 2
          for proj_dim in 32 64 128; do # 32
              for loss in InfoNCE; do # BootstrapLatent JSD BarlowTwins InfoNCE
                python -u models/grace_w_cen.py \
                --model_name 'GRACE_w_cen_me' \
                --seed $seed \
                --gpu $gpu \
                --metric_save_path $metric_save_path \
                --node_data_name $node_data_name \
                --edge_data_name $edge_data_name \
                --cen_feats $cen_feats \
                --lr $lr \
                --input_dim $input_dim \
                --hidden_dim $hidden_dim \
                --gconv_nlayers $gconv_nlayers \
                --proj_dim $proj_dim \
                --loss $loss
              done
            done
          done
        done
      done
  done
done
