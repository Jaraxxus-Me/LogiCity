source /opt/conda/etc/profile.d/conda.sh
conda activate logicity

### suggested EXPNAME rule: <mode>_<max_steps>_fixed_<purpose/feature> ###

### use this EXPNAME only when the final icon lib is finished ###
# EXPNAME="easy_100_fixed"

### use this EXPNAME to compare single icon with variant icons ###
### need to change IMAGE_BASE_PATH in logicity/utils/vis.py to be "./imgs_no_variance" ###
# EXPNAME="easy_100_fixed_no_variance"

### use this EXPNAME for a temp demo ###
EXPNAME="demo_fixed_final"
python3 main.py \
        --config config/tasks/Vis/demo.yaml \
        --mode demo \
        --exp ${EXPNAME} \
        --train_world_num 10 \
        --val_world_num 5 \
        --test_world_num 5 \
        --dataset_dir vis_dataset/${EXPNAME} \
        --max_steps 80 \
        --create_vis_dataset