# source /opt/conda/etc/profile.d/conda.sh
# conda activate logicity

EXPNAME="easy_100_fixed"
python3 main.py \
        --config config/tasks/Vis/easy.yaml \
        --mode easy \
        --exp ${EXPNAME} \
        --train_world_num 10 \
        --val_world_num 2 \
        --test_world_num 2 \
        --dataset_dir vis_dataset/${EXPNAME} \
        --max_steps 100 \
        --create_vis_dataset
