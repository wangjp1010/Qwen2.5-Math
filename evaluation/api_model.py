"""
支持OpenAI API格式（VLLM API）的模型调用
可用于本地或远程API服务器的评测
"""
import requests
import json
import time
from typing import List, Dict, Optional, Any
from tqdm import tqdm


class APIModel:
    """通过OpenAI兼容API调用模型"""

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = None,
        timeout: int = 120,
    ):
        """
        初始化API模型客户端

        Args:
            api_base: API基础地址，默认本地VLLM服务
            api_key: API密钥，VLLM通常使用"EMPTY"
            model_name: 模型名称，如果不提供则从API获取
            timeout: 请求超时时间（秒）
        """
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        # 获取模型信息
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self._get_model_name()

        print(f"Connected to model: {self.model_name}")

    def _get_model_name(self) -> str:
        """获取模型名称"""
        try:
            response = requests.get(
                f"{self.api_base}/models",
                timeout=self.timeout
            )
            if response.status_code == 200:
                models = response.json().get("data", [])
                if models:
                    return models[0].get("id", "unknown")
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
        生成文本

        Args:
            prompts: 提示词列表
            temperature: 温度
            top_p: top-p采样参数
            max_tokens: 最大生成长度
            n: 采样数量
            stop: 停止词列表
            **kwargs: 其他参数

        Returns:
            生成文本列表
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        stop_words = stop or []
        if isinstance(stop_words, str):
            stop_words = [stop_words]

        outputs = []
        batch_size = 32  # VLLM推荐批量大小

        for i in tqdm(range(0, len(prompts), batch_size), desc="API Generating"):
            batch_prompts = prompts[i:i + batch_size]

            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                    for prompt in batch_prompts
                ] if "chat" in self.api_base.lower() or kwargs.get("use_chat_template", False) else batch_prompts,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "n": n,
                "stop": stop_words,
                "stream": False,
            }

            # 移除VLLM不支持的参数
            data = {k: v for k, v in data.items() if v is not None}

            max_retries = 3
            for retry in range(max_retries):
                try:
                    response = requests.post(
                        f"{self.api_base}/chat/completions" if "chat" in self.api_base.lower() or kwargs.get("use_chat_template", False)
                        else f"{self.api_base}/completions",
                        headers=headers,
                        json=data,
                        timeout=self.timeout
                    )

                    if response.status_code == 200:
                        result = response.json()

                        if "chat" in self.api_base.lower() or kwargs.get("use_chat_template", False):
                            batch_outputs = [
                                choice["message"]["content"]
                                for choice in result["choices"]
                            ]
                        else:
                            batch_outputs = [
                                choice["text"]
                                for choice in result["choices"]
                            ]

                        outputs.extend(batch_outputs)
                        break
                    else:
                        error_msg = response.text
                        print(f"API Error (attempt {retry + 1}/{max_retries}): {error_msg}")
                        if retry == max_retries - 1:
                            raise Exception(f"API request failed: {error_msg}")
                        time.sleep(1)

                except requests.exceptions.Timeout:
                    print(f"Request timeout (attempt {retry + 1}/{max_retries})")
                    if retry == max_retries - 1:
                        raise
                    time.sleep(1)
                except Exception as e:
                    print(f"Request error (attempt {retry + 1}/{max_retries}): {e}")
                    if retry == max_retries - 1:
                        raise
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
    ) -> List[str]:
        """
        使用聊天模板生成文本（OpenAI Chat格式）

        Args:
            prompts: 用户提示词列表
            temperature: 温度
            top_p: top-p采样参数
            max_tokens: 最大生成长度
            n: 采样数量
            stop: 停止词列表
            system_prompt: 系统提示词

        Returns:
            生成文本列表
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        stop_words = stop or []

        outputs = []
        batch_size = 32

        for i in tqdm(range(0, len(prompts), batch_size), desc="API Chat Generating"):
            batch_prompts = prompts[i:i + batch_size]

            messages_batch = []
            for prompt in batch_prompts:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                messages_batch.append(messages)

            data = {
                "model": self.model_name,
                "messages": messages_batch,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "n": n,
                "stop": stop_words,
                "stream": False,
            }

            data = {k: v for k, v in data.items() if v is not None}

            max_retries = 3
            for retry in range(max_retries):
                try:
                    response = requests.post(
                        f"{self.api_base}/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=self.timeout
                    )

                    if response.status_code == 200:
                        result = response.json()
                        batch_outputs = [
                            choice["message"]["content"]
                            for choice in result["choices"]
                        ]
                        outputs.extend(batch_outputs)
                        break
                    else:
                        error_msg = response.text
                        print(f"API Error (attempt {retry + 1}/{max_retries}): {error_msg}")
                        if retry == max_retries - 1:
                            raise Exception(f"API request failed: {error_msg}")
                        time.sleep(1)

                except Exception as e:
                    print(f"Request error (attempt {retry + 1}/{max_retries}): {e}")
                    if retry == max_retries - 1:
                        raise
                    time.sleep(1)

        return outputs


def load_api_model(
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    model_name: str = None,
    timeout: int = 120,
) -> APIModel:
    """
    加载API模型

    Args:
        api_base: API基础地址
        api_key: API密钥
        model_name: 模型名称
        timeout: 超时时间

    Returns:
        APIModel实例
    """
    return APIModel(
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        timeout=timeout,
    )


if __name__ == "__main__":
    # 测试API连接
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()

    try:
        model = load_api_model(
            api_base=args.api_base,
            api_key=args.api_key,
            model_name=args.model_name,
        )

        # 测试生成
        test_prompts = ["1+1=", "2+2="]
        outputs = model.generate(test_prompts, max_tokens=10)
        for prompt, output in zip(test_prompts, outputs):
            print(f"Prompt: {prompt}")
            print(f"Output: {output}")
            print("-" * 50)

    except Exception as e:
        print(f"Error: {e}")
