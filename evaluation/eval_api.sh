source /opt/aps/workdir/input/file/pretrain-linear-moe/evaluation/opencompass/venv/bin/activate

export CUDA_VISIBLE_DEVICES="0"
port=3011$CUDA_VISIBLE_DEVICES
export VLLM_API_BASE=http://0.0.0.0:${port}

PROMPT_TYPE="yulan"

bash sh/eval_api.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $VLLM_API_BASE