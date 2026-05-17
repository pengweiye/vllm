from vllm import LLM, SamplingParams

llm = LLM(
    model="/root/autodl-tmp/models/Qwen/Qwen2.5-0.5B-Instruct",
    dtype="auto",
)

params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=32,
)

outputs = llm.generate(
    ["Hello, my name is", "The capital of France is"],
    params,
)

for output in outputs:
    print("=" * 80)
    print("Prompt:", output.prompt)
    print("Output:", output.outputs[0].text)
