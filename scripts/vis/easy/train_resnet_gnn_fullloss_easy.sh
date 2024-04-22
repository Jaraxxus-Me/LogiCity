DATASET_NAME="easy_200"
EXPNAME="resnet_gnn_fullloss"
python3 tools/train_vis_input.py \
    --model LogicityVisPredictorGNN \
    --data_path vis_dataset/${DATASET_NAME} \
    --dataset_name ${DATASET_NAME} \
    --mode easy \
    --exp ${EXPNAME} \
    --epochs 100 \
    --lr 5e-6 \
    --add_concept_loss
