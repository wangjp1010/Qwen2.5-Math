import time
import os
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import openai
from pprint import pformat

def convert_any_to_base_url(any_url: str) -> str:
    """标准化 API 基础地址"""
    if any_url.startswith("http"):
        any_url = any_url
    else:
        any_url = f"http://0.0.0.0:{any_url}"
    
    any_url = any_url.rstrip("/")
    if not any_url.endswith("/v1"):
        any_url = any_url + "/v1"
    return any_url

class APIModel:
    """通过 OpenAI 兼容库调用模型 (支持 vLLM/Local/Remote)"""

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = None,
        timeout: int = 120,
    ):
        # 1. 标准化 URL
        self.api_base = convert_any_to_base_url(api_base)
        self.api_key = api_key
        self.timeout = timeout

        # 2. 初始化 OpenAI 客户端
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout
        )

        # 3. 获取模型信息
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self._get_model_name()

        print(f"Connected to API: {self.api_base}")
        print(f"Using model: {self.model_name}")

    def _get_model_name(self) -> str:
        """从 API 获取第一个可用模型名称"""
        try:
            models = list(self.client.models.list())
            if models:
                return models[0].id
        except Exception as e:
            print(f"Warning: Could not get model name: {e}")
        return "unknown"

    def generate(
        self,
        prompts: List[str],
        temperature: float = 0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        n: int = 1,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        生成文本 (使用 completions 接口)
        """
        stop_words = stop or []
        outputs = []
        batch_size = 1024  # vLLM 支持 batch prompt

        for i in tqdm(range(0, len(prompts), batch_size), desc="API Generating"):
            batch_prompts = prompts[i:i + batch_size]

            max_retries = 3
            for retry in range(max_retries):
                try:
                    # 使用 OpenAI SDK 的 completions 接口
                    # 某些 vLLM 专有参数（如 prompt_logprobs）通过 extra_body 传入
                    response = self.client.completions.create(
                        model=self.model_name,
                        prompt=batch_prompts,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        n=n,
                        stop=stop_words,
                        extra_body=kwargs # 用于透传 prompt_logprobs 等
                    )
                    
                    batch_outputs = [choice.text for choice in response.choices]
                    outputs.extend(batch_outputs)
                    break

                except Exception as e:
                    print(f"Error (attempt {retry + 1}/{max_retries}): {e}")
                    if retry == max_retries - 1:
                        raise e
                    time.sleep(1)

        return outputs

    def generate_with_chat_template(
        self,
        prompts: List[str],
        temperature: float = 0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        n: int = 1,
        stop: Optional[List[str]] = None,
        system_prompt: str = None,
        **kwargs
    ) -> List[str]:
        """
        使用聊天模板生成文本 (使用 chat/completions 接口)
        """
        outputs = []
        
        # 注意：标准 OpenAI SDK 的 chat 接口不支持一次性传入多个 messages 列表
        # 这里的 batch_size 逻辑主要为了进度条显示
        for prompt in tqdm(prompts, desc="API Chat Generating"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            max_retries = 3
            for retry in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        n=n,
                        stop=stop,
                        extra_body=kwargs
                    )
                    outputs.append(response.choices[0].message.content)
                    break
                except Exception as e:
                    print(f"Chat Error (attempt {retry + 1}/{max_retries}): {e}")
                    if retry == max_retries - 1:
                        raise e
                    time.sleep(1)

        return outputs


def load_api_model(
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    model_name: str = None,
    timeout: int = 120,
) -> APIModel:
    return APIModel(
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        timeout=timeout,
    )


if __name__ == "__main__":
    # 模拟简单的命令行测试
    import sys
    
    # 获取参数
    api_url = sys.argv[1] if len(sys.argv) > 1 else "8000" # 兼容端口号或完整URL
    api_key = sys.argv[2] if len(sys.argv) > 2 else "EMPTY"

    try:
        model = load_api_model(api_base=api_url, api_key=api_key)

        # 测试 Completion 模式 (支持 batch)
        test_prompts = ["1+1=", "2+2="]
        print("\nTesting Completions...")
        results = model.generate(test_prompts, max_tokens=10)
        for p, r in zip(test_prompts, results):
            print(f"Q: {p} | A: {r.strip()}")

        # 测试 Chat 模式
        print("\nTesting Chat...")
        chat_results = model.generate_with_chat_template(["你好，请问你是谁？"], max_tokens=50)
        print(f"Chat Output: {chat_results[0]}")

    except Exception as e:
        print(f"Execution failed: {e}")