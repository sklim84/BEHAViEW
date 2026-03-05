seed=2025
gpu='0'

metric_save_path='./_results/exp_results_DGI_transductive.csv'

#node_data_name='HF_FND_2024_03_SAMPLE_100K_NODE_FEAT'
#edge_data_name='HF_FND_2024_03_SAMPLE_100K'
node_data_name='HF_FND_2024_03_SP_H1_NODE_FEAT'
edge_data_name='HF_FND_2024_03_SP_H1'

# "dc" "cc" "bc" "dc cc" "dc bc" "cc bc" "dc cc bc"
for cen_feats in "dc" "cc" "bc" "dc cc" "dc bc" "cc bc" "dc cc bc"; do
  for lr in 0.01 0.001; do # 0.01
    for input_dim in 4 8 16 32; do
      for hidden_dim in 128 256 512 1024; do # 512
        for gconv_nlayers in 3 6 9 12; do # 3
            for loss in JSD; do # BootstrapLatent JSD BarlowTwins InfoNCE
              python -u DGI_transductive_w_cen.py \
              --model_name 'DGI_transductive_w_cen' \
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
