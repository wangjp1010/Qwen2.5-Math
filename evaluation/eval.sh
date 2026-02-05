source /opt/aps/workdir/input/file/pretrain-linear-moe/evaluation/opencompass/venv/bin/activate

PROMPT_TYPE="jiuzhang"

# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="/opt/aps/workdir/input/file/pretrain-linear-moe/cache/models/yulan-team/YuLan-Mini"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH