# source /opt/conda/etc/profile.d/conda.sh
# conda activate logicity

EXPNAME="easy_1k"
python3 main.py \
        --exp ${EXPNAME} \
        --pkl_num 5 \
        --log_dir log_sim \
        --img_dir vis_dataset/${EXPNAME} \
        --dataset_dir vis_dataset \
        --max-steps 1000 \
        --create_vis_dataset