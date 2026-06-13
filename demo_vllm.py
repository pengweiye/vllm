import os
import time
from contextlib import suppress

# Experimental SnapKV prototype switch. Keep this before importing vLLM.
os.environ.setdefault("VLLM_SNAPKV", "1")
os.environ.setdefault("VLLM_SNAPKV_DEBUG", "0")
os.environ.setdefault("VLLM_SNAPKV_DEBUG_LAYERS", "0,31")
os.environ.setdefault("VLLM_SNAPKV_DEBUG_DECODE_STEPS", "3")
os.environ.setdefault("VLLM_SNAPKV_WINDOW_SIZE", "32")
os.environ.setdefault("VLLM_SNAPKV_MAX_CAPACITY", "256")
os.environ.setdefault("VLLM_SNAPKV_KERNEL_SIZE", "7")

from vllm import LLM, SamplingParams


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in ("0", "false", "False", "no", "No")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


model = os.environ.get(
    "VLLM_MODEL",
    "/root/autodl-tmp/models/AI-ModelScope/Mistral-7B-Instruct-v0.2",
)
label = os.environ.get("VLLM_DEMO_LABEL", "snapkv_demo")
max_tokens = _env_int("VLLM_DEMO_MAX_TOKENS", 128)
sections = _env_int("VLLM_DEMO_SECTIONS", 18)
target_prompt_tokens = _env_int("VLLM_DEMO_TARGET_PROMPT_TOKENS", 0)
default_max_num_batched_tokens = 32768 if target_prompt_tokens > 0 else 8192
max_num_batched_tokens = _env_int("VLLM_MAX_NUM_BATCHED_TOKENS",
                                  default_max_num_batched_tokens)
gpu_memory_utilization = _env_float("VLLM_GPU_MEMORY_UTILIZATION", 0.85)
seed = _env_int("VLLM_DEMO_SEED", 0)
benchmark = _env_flag("VLLM_DEMO_BENCHMARK", True)
warmup = _env_flag("VLLM_DEMO_WARMUP", benchmark)
use_tqdm = _env_flag("VLLM_DEMO_TQDM", False)

context_block = """
KV cache compression is useful for long-context inference because transformer
decoding repeatedly attends over the cached keys and values from earlier
tokens. In a long conversation or document, many old tokens become less useful
than the most recent context and a small set of globally important anchor
tokens. SnapKV estimates those important old tokens from the attention pattern
near the end of the prompt, keeps those selected keys and values, and also
keeps the recent observation window. This prototype prompt repeats the same
idea to create a context long enough that selected history, recent tokens, and
generated tokens can be distinguished in debug logs.
""".strip()

FINAL_INSTRUCTION = (
    "\n\nIn one short paragraph, summarize why this compression can reduce "
    "decode-time attention cost while preserving useful context.")


def _make_prompt(num_sections: int) -> str:
    return ("\n\n".join(f"Section {i}: {context_block}"
                       for i in range(1, num_sections + 1)) +
            FINAL_INSTRUCTION)


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def _build_prompt(tokenizer) -> tuple[str, int, int]:
    if target_prompt_tokens <= 0:
        prompt = _make_prompt(sections)
        return prompt, sections, _token_count(tokenizer, prompt)

    sample_sections = max(1, min(sections, 8))
    sample_prompt = _make_prompt(sample_sections)
    sample_tokens = max(1, _token_count(tokenizer, sample_prompt))
    estimated_sections = max(
        1, int((target_prompt_tokens / sample_tokens) * sample_sections))
    current_sections = max(estimated_sections, 1)

    prompt = _make_prompt(current_sections)
    prompt_tokens = _token_count(tokenizer, prompt)

    while prompt_tokens < target_prompt_tokens:
        remaining = target_prompt_tokens - prompt_tokens
        tokens_per_section = max(1, prompt_tokens // current_sections)
        current_sections += max(1, remaining // tokens_per_section)
        prompt = _make_prompt(current_sections)
        prompt_tokens = _token_count(tokenizer, prompt)

    while current_sections > 1:
        candidate = _make_prompt(current_sections - 1)
        candidate_tokens = _token_count(tokenizer, candidate)
        if candidate_tokens < target_prompt_tokens:
            break
        prompt = candidate
        prompt_tokens = candidate_tokens
        current_sections -= 1

    return prompt, current_sections, prompt_tokens


def _make_sampling_params(num_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0,
        max_tokens=num_tokens,
        seed=seed,
    )


def _generated_token_count(output) -> int:
    if not output.outputs:
        return 0
    token_ids = getattr(output.outputs[0], "token_ids", None)
    return len(token_ids) if token_ids is not None else 0


def _run_generate(llm: LLM, prompt: str, num_tokens: int,
                  run_label: str) -> tuple[list, float, int]:
    start = time.perf_counter()
    outputs = llm.generate(
        [prompt],
        _make_sampling_params(num_tokens),
        use_tqdm=use_tqdm,
    )
    elapsed = time.perf_counter() - start
    generated_tokens = _generated_token_count(outputs[0]) if outputs else 0
    print(
        f"Benchmark run={run_label} requested_tokens={num_tokens} "
        f"generated_tokens={generated_tokens} seconds={elapsed:.3f}")
    return outputs, elapsed, generated_tokens

llm = None

try:
    print("=" * 80)
    print("Demo label:", label)
    print("Model:", model)
    print("VLLM_SNAPKV:", os.environ.get("VLLM_SNAPKV"))
    print("VLLM_SNAPKV_MAX_CAPACITY:",
          os.environ.get("VLLM_SNAPKV_MAX_CAPACITY"))
    print("VLLM_SNAPKV_WINDOW_SIZE:", os.environ.get("VLLM_SNAPKV_WINDOW_SIZE"))
    print("VLLM_MAX_NUM_BATCHED_TOKENS:", max_num_batched_tokens)
    print("Target prompt tokens:", target_prompt_tokens or "sections_based")
    print("Initial prompt sections:", sections)
    print("Max output tokens:", max_tokens)
    print("Benchmark mode:", benchmark)
    print("Warmup:", warmup)

    llm = LLM(
        model=model,
        dtype="auto",
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    tokenizer = llm.get_tokenizer()
    prompt, actual_sections, prompt_tokens = _build_prompt(tokenizer)
    print("Prompt sections:", actual_sections)
    print("Prompt tokens:", prompt_tokens)

    if warmup:
        _run_generate(llm, "Warmup request.", 1, "warmup")

    if benchmark:
        _, prefill_probe_elapsed, prefill_probe_tokens = _run_generate(
            llm, prompt, 1, "prefill_probe")
        outputs, full_elapsed, full_generated_tokens = _run_generate(
            llm, prompt, max_tokens, "full_generate")

        decode_estimate_elapsed = max(0.0,
                                      full_elapsed - prefill_probe_elapsed)
        decode_estimate_tokens = max(0,
                                     full_generated_tokens -
                                     prefill_probe_tokens)
        decode_estimate_tps = (
            decode_estimate_tokens / decode_estimate_elapsed
            if decode_estimate_elapsed > 0 else 0.0)
        total_tps = (full_generated_tokens / full_elapsed
                     if full_elapsed > 0 else 0.0)

        print(
            "BENCHMARK_RESULT "
            f"label={label} "
            f"snapkv={os.environ.get('VLLM_SNAPKV')} "
            f"prompt_tokens={prompt_tokens} "
            f"max_tokens={max_tokens} "
            f"generated_tokens={full_generated_tokens} "
            f"prefill_probe_seconds={prefill_probe_elapsed:.3f} "
            f"full_generate_seconds={full_elapsed:.3f} "
            f"decode_estimate_seconds={decode_estimate_elapsed:.3f} "
            f"decode_estimate_tokens={decode_estimate_tokens} "
            f"decode_estimate_tokens_per_second={decode_estimate_tps:.3f} "
            f"total_tokens_per_second={total_tps:.3f}")
        elapsed = full_elapsed
    else:
        outputs, elapsed, _ = _run_generate(llm, prompt, max_tokens,
                                            "single_generate")

    for output in outputs:
        print("=" * 80)
        print("Prompt chars:", len(output.prompt))
        print("Prompt tokens:", prompt_tokens)
        print("Prompt preview:", output.prompt[:240].replace("\n", " "), "...")
        print(f"Elapsed seconds: {elapsed:.3f}")
        print("Output:", output.outputs[0].text)
finally:
    # V1 LLM uses a background EngineCore process. The LLM class does not
    # expose a public shutdown method, so explicitly close the core client in
    # this debug demo to avoid a noisy monitor-thread warning at interpreter
    # shutdown.
    if llm is not None:
        with suppress(Exception):
            llm.llm_engine.engine_core.shutdown()
