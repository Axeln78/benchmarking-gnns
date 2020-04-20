#!/bin/bash

# bash script_main_CSL_graph_classification.sh


############
# GNNs
############

#GatedGCN
#GCN
#GraphSage
#MLP
#GIN
#MoNet
#GAT
#DiffPool


code=main_CSL_graph_classification.py 
dataset=CSL
tmux new -s benchmark_CSL_graph_classification -d
tmux send-keys "source activate benchmark_gnn" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --config 'configs/CSL_graph_classification_MLP.json' &
python $code --dataset $dataset --gpu_id 1 --config 'configs/CSL_graph_classification_MLP_GATED.json' &
python $code --dataset $dataset --gpu_id 2 --config 'configs/CSL_graph_classification_GCN.json' &
python $code --dataset $dataset --gpu_id 3 --config 'configs/CSL_graph_classification_GAT.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --config 'configs/CSL_graph_classification_GraphSage.json' &
python $code --dataset $dataset --gpu_id 1 --config 'configs/CSL_graph_classification_GIN.json' &
python $code --dataset $dataset --gpu_id 2 --config 'configs/CSL_graph_classification_MoNet.json' &
python $code --dataset $dataset --gpu_id 3 --config 'configs/CSL_graph_classification_DiffPool.json' &
wait" C-m
tmux send-keys "tmux kill-session -t benchmark_CSL_graph_classification" C-m











