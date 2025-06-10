# source /opt/conda/etc/profile.d/conda.sh
conda activate logicity
EXPNAME="expert_100"
MAXSETP=100

for s in 0
do
python3 main.py --config "config/tasks/sim/expert.yaml" \
        --exp ${EXPNAME}_${s} --max-steps $MAXSETP --seed $s \
        --log_dir log_sim
done