source /opt/conda/etc/profile.d/conda.sh
conda activate logicity

### suggested EXPNAME rule: <mode>_<max_steps>_fixed_<purpose/feature> ###

### use this EXPNAME only when the final icon lib is finished ###
# EXPNAME="easy_100_fixed"

### use this EXPNAME to compare single icon with variant icons ###
### need to change IMAGE_BASE_PATH in logicity/utils/vis.py to be "./imgs_no_variance" ###
# EXPNAME="easy_100_fixed_no_variance"   

### use this EXPNAME for a temp demo ###
EXPNAME="easy_100_fixed_demo"
python3 main.py \
        --config config/tasks/Vis/easy.yaml \
        --mode easy \
        --exp ${EXPNAME} \
        --train_world_num 40 \
        --val_world_num 10 \
        --test_world_num 10 \
        --dataset_dir vis_dataset/${EXPNAME} \
        --max_steps 100 \
        --create_vis_dataset