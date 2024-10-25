source /opt/conda/etc/profile.d/conda.sh
conda activate base

export AZURE_OPENAI_API_KEY=da81896b9cf74c51bb971c82f5bf5b0f
export AZURE_OPENAI_API_BASE=https://zhaoyu.openai.azure.com/
export AZURE_OPENAI_API_VERSION=2024-04-01-preview

python3 tools/test_mmlu_gpt.py --shots 5 --exp gpt4t_5shot_gp --start 4800 --end 9600 --good_prompt