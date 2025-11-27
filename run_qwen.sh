export MASTER_PORT=$(python -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export APP_SEED="42"
export NUM_GPUS=8
# Set CUDA_VISIBLE_DEVICES to use specific GPUs (uncomment and modify as needed)
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# WANDB_PROJECT=spo-math APP_EXPERIMENT_NAME=qwen2.5_7b-spo-tree-666 APP_DIRECTORY=SPO-experiments APP_MINIMIZE_STORED_FILES=True deepspeed --no_local_rank --num_gpus=$NUM_GPUS --master_port $MASTER_PORT --include localhost:0  src/treetune/main.py --configs "configs/qwen_2_5_7b_math_spo_chain_MATH.jsonnet" run_iteration_loop

WANDB_PROJECT=spo-math APP_EXPERIMENT_NAME=qwen2.5_7b-spo-tree-666 APP_DIRECTORY=SPO-experiments APP_MINIMIZE_STORED_FILES=True deepspeed --no_local_rank --num_gpus=$NUM_GPUS --master_port $MASTER_PORT src/treetune/main.py --configs "configs/qwen_2_5_7b_math_spo_chain_MATH.jsonnet" run_iteration_loop