source /opt/conda/etc/profile.d/conda.sh
conda activate logicity

EXPNAME="very_easy_random_final"
python3 main.py \
        --config config/tasks/Vis/very_easy.yaml \
        --mode easy \
        --exp ${EXPNAME} \
        --train_world_num 100 \
        --val_world_num 20 \
        --test_world_num 20 \
        --dataset_dir vis_dataset/${EXPNAME} \
        --max_steps 100 \
        --min_agent_num_train 12 \
        --max_agent_num_train 12 \
        --min_agent_num_val 13 \
        --max_agent_num_val 15 \
        --min_agent_num_test 13 \
        --max_agent_num_test 15 \
        --create_vis_dataset
