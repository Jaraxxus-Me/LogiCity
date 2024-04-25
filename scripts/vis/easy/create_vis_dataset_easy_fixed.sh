# source /opt/conda/etc/profile.d/conda.sh
# conda activate logicity

EXPNAME="easy_200_fixed"
python3 main.py \
        --config config/tasks/Vis/easy.yaml \
        --mode easy \
        --exp ${EXPNAME} \
        --train_world_num 20 \
        --val_world_num 5 \
        --test_world_num 5 \
        --dataset_dir vis_dataset/${EXPNAME} \
        --max_steps 200 \
        --create_vis_dataset
