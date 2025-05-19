datapath=data/mvtec
datasets=('bottle'  'cable'  'capsule'   'hazelnut'  'metal_nut'  'pill' 'screw'  'toothbrush' 'transistor' 'zipper' )
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

############# Detection
python bin/run.py --gpu 0 --seed 26 \
--log_group IM336_S2_P010_0075_E100_100_rho75_kfg30_kbg10_L100_Q025_26  --log_project MVTec_rho_Results_0411 results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 \
--k_fg 30 --k_bg 10 --query_p 0.25 --bg_p 0.075 --rho 0.5 --length 100 --patchsize 3 --anomaly_scorer_num_nn 1  \
--feature_path "data/vis_features/online_uni_feature_mask_K2_L15_AN1" \
sampler -p 0.10 approx_greedy_coreset \
dataset --resize 368 --imagesize 336 --shot 2  "${dataset_flags[@]}" mvtec $datapath
