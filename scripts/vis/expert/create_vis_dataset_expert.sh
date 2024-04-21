# source /opt/conda/etc/profile.d/conda.sh
# conda activate logicity

EXPNAME="expert_200"
python3 main.py \
        --config config/tasks/sim/expert.yaml \
        --exp ${EXPNAME} \
        --train_world_num 20 \
        --val_world_num 5 \
        --test_world_num 5 \
        --dataset_dir vis_dataset/${EXPNAME} \
        --max_steps 200 \
        --min_agent_num_train 11 \
        --max_agent_num_train 13 \
        --min_agent_num_val 11 \
        --max_agent_num_val 13 \
        --min_agent_num_test 14 \
        --max_agent_num_test 18 \
        --create_vis_dataset
