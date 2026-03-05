seed=2025
gpu='0'

metric_save_path='./_results/exp_results_GBT_rq3.csv'

node_data_name='HF_FND_2024_03_SAMPLE_100K_NODE_FEAT'
edge_data_name='HF_FND_2024_03_SAMPLE_100K'

# "dc" "cc" "bc" "dc cc" "dc bc" "cc bc" "dc cc bc"
for cen_feats in "dc" "cc" "bc" "dc cc" "dc bc" "cc bc" "dc cc bc"; do
  for lr in 5e-4; do # 5e-4
     for input_dim in 16; do
      for hidden_dim in 128; do # 256
            for loss in BarlowTwins; do # BarlowTwins
              python -u GBT_w_cen.py \
              --model_name 'GBT_w_cen' \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --node_data_name $node_data_name \
              --edge_data_name $edge_data_name \
              --cen_feats $cen_feats \
              --lr $lr \
              --input_dim $input_dim \
              --hidden_dim $hidden_dim \
              --loss $loss
            done
          done
        done
      done
  done
done
