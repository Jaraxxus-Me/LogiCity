# source /opt/conda/etc/profile.d/conda.sh
# conda activate logicity

EXPNAME="easy_1k"
LOGNAME="log_sim"
for s in 0 1 2 3 4
do
python3 tools/pkl2city.py \
    --pkl ${LOGNAME}/${EXPNAME}_${s}.pkl \
    --ego_id -1 \
    --output_folder vis_dataset/${EXPNAME}/${EXPNAME}_${s}_imgs
done
