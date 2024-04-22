DATASET_NAME="expert_200"
EXPNAME="resnet_nlm"
python3 tools/train_vis_input.py \
    --model LogicityVisPredictorNLM \
    --data_path vis_dataset/${DATASET_NAME} \
    --dataset_name ${DATASET_NAME} \
    --mode expert \
    --exp ${EXPNAME}_${DATASET_NAME} \
    --epochs 100 \
    --lr 5e-4
