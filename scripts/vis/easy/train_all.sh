source /opt/conda/etc/profile.d/conda.sh
conda activate logicity


python3 tools/train_vis_input.py --config config/tasks/Vis/ResNetNLM/easy_100_fixed_demo_modular.yaml \
    --exp easy_100_fixed_demo_nlm_modular \
    --modular >> log_sim/easy_100_fixed_demo_nlm_modular.log

python3 tools/train_vis_input.py --config config/tasks/Vis/ResNetNLM/easy_100_fixed_demo_e2e.yaml \
    --exp easy_100_fixed_demo_nlm_e2e >> log_sim/easy_100_fixed_demo_nlm_e2e.log

python3 tools/train_vis_input.py --config config/tasks/Vis/ResNetNLM/easy_100_fixed_no_variance_modular.yaml \
    --exp easy_100_fixed_no_variance_nlm_modular \
    --modular >> log_sim/easy_100_fixed_no_variance_nlm_modular.log

python3 tools/train_vis_input.py --config config/tasks/Vis/ResNetNLM/easy_100_fixed_no_variance_e2e.yaml \
    --exp easy_100_fixed_no_variance_nlm_e2e >> log_sim/easy_100_fixed_no_variance_nlm_e2e.log

python3 tools/train_vis_input_bilevel.py --config config/tasks/Vis/ResNetNLM/easy_100_fixed_demo_bilevel.yaml \
    --exp easy_100_fixed_demo_bilevel_unrolled >> log_sim/easy_100_fixed_demo_bilevel_unrolled.log

python3 tools/train_vis_input_bilevel.py --config config/tasks/Vis/ResNetNLM/easy_100_fixed_demo_bilevel.yaml \
    --exp easy_100_fixed_demo_bilevel_implicit \
    --implicit >> log_sim/easy_100_fixed_demo_bilevel_implicit.log