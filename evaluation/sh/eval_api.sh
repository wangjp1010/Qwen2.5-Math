set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
API_BASE=$3
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME="gsm8k,math_oai,gsm_hard,svamp,gaokao2023en,olympiadbench,college_math"
# DATA_NAME="gsm_hard,svamp,minerva_math,gaokao2023en,olympiadbench,college_math"
TOKENIZERS_PARALLELISM=false \
python eval_api.py \
    --api_base $API_BASE \
    --api_model_name $MODEL_NAME_OR_PATH \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --max_tokens_per_call 8192 \
