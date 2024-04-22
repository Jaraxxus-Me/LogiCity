# source /opt/conda/etc/profile.d/conda.sh
# conda activate logicity

EXPNAME="expert_200_fixed"
python3 main.py \
        --config config/tasks/sim/expert.yaml \
        --mode expert \
        --exp ${EXPNAME} \
        --train_world_num 20 \
        --val_world_num 5 \
        --test_world_num 5 \
        --dataset_dir vis_dataset/${EXPNAME} \
        --max_steps 200 \
        --create_vis_dataset
