# source /opt/conda/etc/profile.d/conda.sh
# conda activate logicity

EXPNAME="hard_500_fixed"
python3 main.py \
        --config config/tasks/Vis/hard.yaml \
        --mode hard \
        --exp ${EXPNAME} \
        --train_world_num 1 \
        --val_world_num 1 \
        --test_world_num 1 \
        --dataset_dir vis_dataset/${EXPNAME} \
        --max_steps 500 \
        --create_vis_dataset
