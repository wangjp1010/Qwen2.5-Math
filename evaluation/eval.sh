source /opt/aps/workdir/input/file/pretrain-linear-moe/evaluation/opencompass/venv/bin/activate

PROMPT_TYPE="yulan"

# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="1"
MODEL_NAME_OR_PATH="/opt/aps/workdir/input/file/pretrain-linear-moe/megatron_lm_workspace/checkpoint/Distilled-s4096-mathcode10b-s1randg-sch1-CPT-test100b4b-nodecay-CPT-wjp-ablation_reasoning1-GDN2.9b-nl56-hs1920-mtp0-la4-kh64-vh64-nkh8-nvh32-ah30qg6A7-12_21_22_23_46_48_49-mp2pp1cp2-sl32768bs128lr2e5mlr7e7/iter_714-hf"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH