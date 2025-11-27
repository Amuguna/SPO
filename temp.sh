export MASTER_PORT=$(python -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export APP_SEED="42"

WANDB_PROJECT=spo-math APP_EXPERIMENT_NAME=qwen1.5b-spo-tree-666 APP_DIRECTORY=SPO-experiments APP_MINIMIZE_STORED_FILES=True deepspeed --master_port $MASTER_PORT --include localhost:0  src/treetune/main.py --configs "configs/polIter_qwen1b_spo_tree_MATH.jsonnet,configs/gpus/gpu_0.jsonnet" run_iteration_loop