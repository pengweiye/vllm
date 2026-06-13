import os
import subprocess
import sys
import time


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_capacities() -> list[int]:
    return _parse_int_list("VLLM_SNAPKV_COMPARE_CAPACITIES", "256")


def _parse_int_list(name: str, default: str) -> list[int]:
    raw = os.environ.get(name, default)
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            raise ValueError(
                f"{name} must be comma-separated ints, got {raw!r}") from None
    return values


def _run_case(label: str, overrides: dict[str, str]) -> int:
    env = os.environ.copy()
    env.update(overrides)
    env.setdefault("PYTHONUNBUFFERED", "1")

    print("\n" + "=" * 96, flush=True)
    print(f"Running case: {label}", flush=True)
    print("=" * 96, flush=True)

    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "demo_vllm.py"],
        env=env,
        check=False,
    )
    elapsed = time.perf_counter() - start

    print("=" * 96, flush=True)
    print(
        f"Case finished: {label} exit_code={proc.returncode} "
        f"wall_seconds={elapsed:.3f}",
        flush=True,
    )
    return proc.returncode


def main() -> int:
    run_baseline = os.environ.get("VLLM_SNAPKV_COMPARE_BASELINE",
                                  "1") not in ("0", "false", "False")
    prompt_token_targets = _parse_int_list("VLLM_SNAPKV_COMPARE_PROMPT_TOKENS",
                                           "8192,16384")
    output_token_targets = _parse_int_list("VLLM_SNAPKV_COMPARE_OUTPUT_TOKENS",
                                           "256,512")
    window_size = os.environ.get("VLLM_SNAPKV_WINDOW_SIZE", "32")
    kernel_size = os.environ.get("VLLM_SNAPKV_KERNEL_SIZE", "7")
    max_num_batched_tokens = os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS",
                                            "32768")

    exit_codes: list[int] = []

    for prompt_tokens in prompt_token_targets:
        for output_tokens in output_token_targets:
            common_env = {
                "VLLM_DEMO_BENCHMARK": "1",
                "VLLM_DEMO_WARMUP": "1",
                "VLLM_DEMO_TQDM": "0",
                "VLLM_DEMO_TARGET_PROMPT_TOKENS": str(prompt_tokens),
                "VLLM_DEMO_MAX_TOKENS": str(output_tokens),
                "VLLM_MAX_NUM_BATCHED_TOKENS": max_num_batched_tokens,
            }

            if run_baseline:
                label = f"baseline_p{prompt_tokens}_o{output_tokens}"
                exit_codes.append(
                    _run_case(
                        label,
                        {
                            **common_env,
                            "VLLM_DEMO_LABEL": label,
                            "VLLM_SNAPKV": "0",
                            "VLLM_SNAPKV_DEBUG": "0",
                        },
                    ))

            for capacity in _parse_capacities():
                label = (f"snapkv_p{prompt_tokens}_o{output_tokens}"
                         f"_c{capacity}")
                exit_codes.append(
                    _run_case(
                        label,
                        {
                            **common_env,
                            "VLLM_DEMO_LABEL": label,
                            "VLLM_SNAPKV": "1",
                            "VLLM_SNAPKV_DEBUG": "0",
                            "VLLM_SNAPKV_WINDOW_SIZE": window_size,
                            "VLLM_SNAPKV_MAX_CAPACITY": str(capacity),
                            "VLLM_SNAPKV_KERNEL_SIZE": kernel_size,
                        },
                    ))

    failed = [code for code in exit_codes if code != 0]
    if failed:
        print(f"Some cases failed: {failed}", flush=True)
        return 1

    print("\nAll comparison cases finished successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
