# API Evaluation for Qwen2.5-Math

本模块支持通过OpenAI API（VLLM API）格式进行数学问题评测，无需本地GPU。

## 前置条件

### 1. 启动VLLM API服务

**方式一：使用Docker**
```bash
docker run --gpus all -p 8000:8000 \
  -v /path/to/models:/models \
  vllm/vllm-openai:latest \
  --model /models/Qwen2.5-Math-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

**方式二：直接运行**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code
```

### 2. 测试API连接

```bash
curl http://localhost:8000/v1/models
```

## 使用方法

### 基本用法

```bash
cd evaluation

# 设置API地址
export API_BASE="http://localhost:8000/v1"

# 运行评测
python eval_api.py \
  --data_names gsm8k,math \
  --api_base $API_BASE \
  --api_model_name Qwen2.5-Math-7B-Instruct \
  --prompt_type tool-integrated \
  --num_test_sample 100 \
  --temperature 0 \
  --save_outputs
```

### 完整参数

```bash
python eval_api.py \
  --data_names gsm8k,math,cmath \
  --data_dir ./data \
  --output_dir ./output \
  --api_base "http://localhost:8000/v1" \
  --api_model_name Qwen2.5-Math-7B-Instruct \
  --api_key "EMPTY" \
  --prompt_type tool-integrated \
  --split test \
  --num_test_sample 200 \
  --seed 0 \
  --start 0 \
  --end -1 \
  --temperature 0 \
  --n_sampling 1 \
  --top_p 1 \
  --max_tokens_per_call 2048 \
  --shuffle \
  --save_outputs \
  --overwrite \
  --use_chat_template \
  --system_prompt "Please reason step by step, and put your final answer within \\boxed{}." \
  --api_timeout 120 \
  --batch_size 32
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api_base` | `http://localhost:8000/v1` | VLLM API基础地址 |
| `--api_model_name` | `None` | 模型名称（自动检测） |
| `--api_key` | `EMPTY` | API密钥 |
| `--use_chat_template` | `True` | 是否使用聊天模板 |
| `--system_prompt` | `step by step...` | 系统提示词 |
| `--api_timeout` | `120` | API超时时间（秒） |
| `--batch_size` | `32` | 批量请求大小 |
| `--prompt_type` | `tool-integrated` | 推理模式：cot, pal, tool-integrated |
| `--temperature` | `0` | 采样温度（0=贪婪） |
| `--n_sampling` | `1` | 采样次数 |
| `--num_test_sample` | `-1` | 测试样本数（-1=全部） |

## 提示词类型

- **cot**: 思维链模式
- **pal**: 程序辅助语言模型
- **tool-integrated**: 工具集成推理（推荐）

## 输出格式

评测结果保存为JSONL文件，每行一个样本：

```json
{
  "idx": 0,
  "question": "Find the value of $x$ that satisfies...",
  "gt": "42",
  "gt_cot": "step by step...",
  "code": "...",
  "pred": ["42"],
  "report": "...",
  "score": [true]
}
```

## 分布式API

如果使用多个API服务，可以使用负载均衡：

```python
from api_model import APIModel

models = [
    APIModel(api_base="http://gpu1:8000/v1"),
    APIModel(api_base="http://gpu2:8000/v1"),
]
```

## 注意事项

1. API服务器需要支持 `chat/completions` 端点
2. 建议使用 `trust_remote_code=True` 以支持Qwen模型
3. 大批量评测时注意API限流
4. 使用 `--overwrite` 参数会重新评测已处理的样本
