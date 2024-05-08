source /opt/conda/etc/profile.d/conda.sh
conda activate logicity

EXPNAME="hard_fixed_final"
python3 main.py \
        --config config/tasks/Vis/hard.yaml \
        --mode hard \
        --exp ${EXPNAME} \
        --train_world_num 100 \
        --val_world_num 20 \
        --test_world_num 20 \
        --dataset_dir vis_dataset/${EXPNAME} \
        --max_steps 100 \
        --create_vis_dataset
