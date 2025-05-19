datapath=/media/x/4t/zhy/datasets/mpdd
datasets=('bracket_black' 'bracket_brown' 'bracket_white' 'connector' 'metal_plate'  'tubes')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

############# Detection
python bin/run.py --gpu 0 --seed 26  \
--log_group test  --log_project MPDD_WRN50_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 \
--k_fg 10 --k_bg 40 --query_p 0.25 --bg_p 0.75 --rho 0.5 --length 100 --patchsize 3 --anomaly_scorer_num_nn 1  \
--feature_path "data/vis_features/online_uni_feature_mask_K2_L15_AN1" \
sampler -p 0.15 approx_greedy_coreset \
dataset --resize 368 --imagesize 336 --shot 1 "${dataset_flags[@]}" mpdd $datapath