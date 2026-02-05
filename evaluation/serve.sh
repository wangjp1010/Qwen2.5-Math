source /opt/aps/workdir/input/file/pretrain-linear-moe/evaluation/opencompass/venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
port=3011$CUDA_VISIBLE_DEVICES

all_hf_path=(
    /opt/aps/workdir/input/file/pretrain-linear-moe/wangjiapeng/qy/Llama-3.2-3B-Instruct
)    

hf_path=${all_hf_path[$CUDA_VISIBLE_DEVICES]}
echo "Using model path: $hf_path"

# VLLM_ATTENTION_BACKEND="FLASHINFER" \
VLLM_USE_V1="1" \
VLLM_ENGINE=1 \
vllm serve $hf_path \
    --port $port \
    --max-num-seqs 1024 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1 \
    --trust-request-chat-template