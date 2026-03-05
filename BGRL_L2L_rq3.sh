seed=2025
gpu='0'

metric_save_path='./_results/exp_results_BGRL_L2L_rq3.csv'

#node_data_name='HF_FND_2024_03_SAMPLE_100K_NODE_FEAT'
#edge_data_name='HF_FND_2024_03_SAMPLE_100K'
node_data_name='HF_FND_2024_03_SP_H1_NODE_FEAT'
edge_data_name='HF_FND_2024_03_SP_H1'


for cen_feats in "dc" "cc" "bc" "dc cc" "dc bc" "cc bc" "dc cc bc"; do
  for lr in 0.1; do # 0.01
    for input_dim in 16; do
      for hidden_dim in 512; do # 256
        for gconv_nlayers in 1; do # 2
            for loss in BarlowTwins; do
              python -u BGRL_L2L_w_cen.py \
              --model_name 'BGRL_L2L_w_cen' \
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
              --loss $loss
            done
          done
        done
      done
    done
  done
done
