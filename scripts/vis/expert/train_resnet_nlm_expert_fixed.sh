DATASET_NAME="expert_200_fixed"
EXPNAME="resnet_nlm_fixed"
python3 tools/train_vis_input.py \
    --model LogicityVisPredictorNLM \
    --data_path vis_dataset/${DATASET_NAME} \
    --dataset_name ${DATASET_NAME} \
    --mode expert \
    --exp ${EXPNAME} \
    --epochs 100 \
    --lr 5e-4