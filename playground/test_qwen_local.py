from vllm import LLM, SamplingParams

MODEL_DIR = "/root/autodl-tmp/models/Qwen/Qwen2.5-0.5B-Instruct"

llm = LLM(
    model=MODEL_DIR,
    dtype="float16",
    max_model_len=1024,
    max_num_seqs=1,
    gpu_memory_utilization=0.75,
    enforce_eager=True,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
)

outputs = llm.generate(
    ["用三句话解释什么是 KV cache。"],
    sampling_params,
)

print("\n===== OUTPUT =====\n")
print(outputs[0].outputs[0].text)
