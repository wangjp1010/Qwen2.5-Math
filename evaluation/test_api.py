#!/usr/bin/env python3
"""
快速测试API连接和生成功能
"""

from api_model import APIModel, load_api_model
import argparse


def test_api_connection(api_base, api_key, model_name, use_chat_template):
    """测试API连接"""
    print("=" * 60)
    print("Testing API Connection")
    print("=" * 60)

    try:
        model = load_api_model(
            api_base=api_base,
            api_key=api_key,
            model_name=model_name,
        )
        print(f"✓ Connected to model: {model.model_name}")
        return model
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return None


def test_generation(model, use_chat_template):
    """测试生成功能"""
    if model is None:
        return

    print("\n" + "=" * 60)
    print("Testing Generation")
    print("=" * 60)

    prompts = [
        "Find the value of x: 2x + 5 = 15",
        "What is 15% of 80?",
        "Solve: 3(x - 2) = 12",
    ]

    stop_words = ["\n\n", "```"]

    try:
        if use_chat_template:
            outputs = model.generate_with_chat_template(
                prompts=prompts,
                temperature=0,
                max_tokens=100,
                system_prompt="Please reason step by step and put your final answer in \\boxed{}.",
                stop=stop_words,
            )
        else:
            outputs = model.generate(
                prompts=prompts,
                temperature=0,
                max_tokens=100,
                stop=stop_words,
            )

        print("\nResults:")
        for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
            print(f"\n{i}. Prompt: {prompt}")
            print(f"   Output: {output[:200]}..." if len(output) > 200 else f"   Output: {output}")
            print("-" * 40)

    except Exception as e:
        print(f"✗ Generation failed: {e}")


def test_batch_generation(model, use_chat_template):
    """测试批量生成"""
    if model is None:
        return

    print("\n" + "=" * 60)
    print("Testing Batch Generation (10 prompts)")
    print("=" * 60)

    prompts = [f"Calculate: {i} + {i*2} =" for i in range(1, 11)]

    try:
        if use_chat_template:
            outputs = model.generate_with_chat_template(
                prompts=prompts,
                temperature=0,
                max_tokens=50,
                system_prompt="Give a brief answer.",
            )
        else:
            outputs = model.generate(
                prompts=prompts,
                temperature=0,
                max_tokens=50,
            )

        success_count = sum(1 for _ in outputs)
        print(f"\n✓ Successfully generated {success_count}/{len(prompts)} responses")

    except Exception as e:
        print(f"✗ Batch generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test API Model")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1",
                        help="API base URL")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API key")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name")
    parser.add_argument("--use_chat_template", action="store_true", default=True,
                        help="Use chat template")

    args = parser.parse_args()

    model = test_api_connection(args.api_base, args.api_key, args.model_name, args.use_chat_template)

    if model:
        test_generation(model, args.use_chat_template)
        test_batch_generation(model, args.use_chat_template)

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)


if __name__ == "__main__":
    main()
