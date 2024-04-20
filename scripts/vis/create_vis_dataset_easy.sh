# source /opt/conda/etc/profile.d/conda.sh
# conda activate logicity

EXPNAME="easy_200"
python3 main.py \
        --exp ${EXPNAME} \
        --train_world_num 5 \
        --val_world_num 1 \
        --test_world_num 1 \
        --dataset_dir vis_dataset/${EXPNAME} \
        --max_steps 200 \
        --min_agent_num 5 \
        --max_agent_num 8 \
        --create_vis_dataset
