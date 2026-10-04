#!/usr/bin/env python3
"""Evaluate vanilla decoding, Spiffy and SpecFold on one (model, benchmark) pair.

Prints NFE, TPS and accuracy per method (one entry of Table 1), writes one result
JSON per method and a summary JSON under --output_dir/<model>/<benchmark>/.
The draft graph for (model, B, D, S) is loaded from calibration_graph/ or
calibrated on first use.

    python run_eval.py --model nld_3b --dataset gsm8k
    python run_eval.py --model fastdllm_v2_7b --dataset humaneval --methods spiffy specfold --delta 0.2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from specfold.calibration.calibration_sets import CALIBRATION_SETS, calibration_set_size
from specfold.calibration.runner import load_or_build_calibration_graph
from specfold.config import Config
from specfold.evaluate.evalplus_scoring import EVALPLUS_ACCURACY_KEY, EVALPLUS_DATASETS, run_evalplus_scoring
from specfold.evaluate.lmeval_adapter import (
    DLLMEvalAdapter,
    boxed_answer_postprocessor,
    compact_lmeval_result,
    ensure_lmeval_runtime_env,
    fastdllm_v2_official_math_prompt_rewrite,
    get_lmeval_dataset_config,
    patch_minerva_math_process_results,
    primary_metric_from_results,
)
from specfold.generate.fastdllm_model_executor import FastDLLMModelExecutor
from specfold.generate.fastdllm_v2_folded import SpecFoldTritonFastDLLMV2Generation
from specfold.generate.fastdllm_v2_generation import FastDLLMV2SpeculativeGeneration
from specfold.generate.nld_folded import SpecFoldTritonNLDGeneration
from specfold.generate.nld_generation import NLDDiffusionGeneration
from specfold.generate.nld_model_executor import NLDModelExecutor
from specfold.registry import DATASET_NAMES, MODEL_SPECS, NLD_MODELS, get_model_spec, normalize_dataset_name
from specfold.utils import current_gpu_name, resolve_fastdllm_v2_model_path, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

METHODS = ("vanilla", "spiffy", "specfold")
MODELS = tuple(MODEL_SPECS)

@dataclass
class GenerationStats:
    output: str
    tokens_generated: int
    model_calls: int
    drafts_computed: int
    elapsed_s: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vanilla / Spiffy / SpecFold on one (model, benchmark) pair")
    p.add_argument("--model", default="nld_3b", choices=MODELS)
    p.add_argument("--dataset", default="gsm8k", help=f"one of {', '.join(DATASET_NAMES)}")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--num_samples", type=int, default=None,
                   help="evaluate a prefix of the benchmark (not for humaneval/mbpp, which EvalPlus scores whole)")
    p.add_argument("--num_fewshot", type=int, default=None, help="default: 0-shot for every benchmark")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="cuda, cuda:N, cpu, or auto")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--pretrained_path", default=None, help="override the HF checkpoint id/path")
    p.add_argument("--log_level", default="INFO")

    dec = p.add_argument_group("decoding")
    dec.add_argument("--block_length", type=int, default=Config.BLOCK_LENGTH)
    dec.add_argument("--small_block_size", type=int, default=8, help="Fast-dLLM v2 sub-block size")
    dec.add_argument("--threshold", type=float, default=0.9, help="unmasking confidence threshold (not delta)")
    dec.add_argument("--temperature", type=float, default=0.0)
    dec.add_argument("--top_p", type=float, default=0.95)
    dec.add_argument("--top_k", type=int, default=None)
    dec.add_argument("--max_new_tokens", type=int, default=2048)
    dec.add_argument("--max_length", type=int, default=4096)
    dec.add_argument("--use_block_cache", action="store_true", default=False, help="Fast-dLLM v2 block KV cache")
    dec.add_argument("--no_apply_chat_template", dest="apply_chat_template", action="store_false", default=True)
    dec.add_argument("--no_fewshot_as_multiturn", dest="fewshot_as_multiturn", action="store_false", default=True)

    graph = p.add_argument_group("draft graph (spiffy, specfold)")
    graph.add_argument("--budget_B", type=int, default=Config.BUDGET_B, help="draft budget B (graph nodes)")
    graph.add_argument("--depth_D", type=int, default=Config.DEPTH_D, help="graph depth D")
    graph.add_argument("--per_step_limit", type=int, default=Config.PER_STEP_LIMIT, help="per-level limit S")
    graph.add_argument("--calibration_set", default=Config.CALIBRATION_SET, choices=sorted(CALIBRATION_SETS))
    graph.add_argument("--calibration_suffix", default="", help="appended to the graph filename")

    fold = p.add_argument_group("specfold")
    fold.add_argument("--delta", type=float, default=0.1, help="recompute threshold on the relative residual")
    args = p.parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    return args


def result_label(method: str, args: argparse.Namespace) -> str:
    """vanilla | spiffy_B<B>_D<D>_S<S> | specfold_d<delta>_B<B>_D<D>_S<S>"""
    if method == "vanilla":
        return "vanilla"
    label = "spiffy" if method == "spiffy" else "specfold_d" + f"{args.delta:g}".replace(".", "p")
    return f"{label}_B{args.budget_B}_D{args.depth_D}_S{args.per_step_limit}"


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device)


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "bfloat16" and (device.type != "cuda" or torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    if dtype == "float16" and device.type == "cuda":
        return torch.float16
    return torch.float32


def load_executor(model_key: str, device: torch.device, dtype: torch.dtype, pretrained_path: Optional[str]):
    """Load the checkpoint once; every method's generator shares it."""
    if model_key in NLD_MODELS:
        path = pretrained_path or MODEL_SPECS[model_key].pretrained_path
        logger.info("Loading %s from %s on %s", model_key, path, device)
        return NLDModelExecutor(pretrained_path=path, device=device, target_dtype=dtype)
    path = pretrained_path or resolve_fastdllm_v2_model_path(model_variant=MODEL_SPECS[model_key].model_variant)
    logger.info("Loading %s from %s on %s", model_key, path, device)
    return FastDLLMModelExecutor(pretrained_path=path, device=device, target_dtype=dtype)


def build_generator(method: str, model_key: str, executor: Any, graph: Optional[Dict[str, Any]],
                    device: torch.device, args: argparse.Namespace) -> Any:
    is_nld = model_key in NLD_MODELS
    if method == "specfold":
        gen_cls = SpecFoldTritonNLDGeneration if is_nld else SpecFoldTritonFastDLLMV2Generation
        extra = dict(delta=args.delta)
    else:
        gen_cls = NLDDiffusionGeneration if is_nld else FastDLLMV2SpeculativeGeneration
        extra = {}
    common = dict(
        block_length=args.block_length,
        model_executor=executor,
        calibration_graph=graph if method != "vanilla" else None,
        temperature=args.temperature,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        device=str(device),
    )
    if is_nld:
        return gen_cls(**common, **extra)
    return gen_cls(
        **common,
        top_p=args.top_p,
        top_k=args.top_k,
        small_block_size=args.small_block_size,
        use_block_cache=args.use_block_cache,
        dtype=args.dtype,
        **extra,
    )


def make_generate_fn(generator: Any) -> Any:
    def generate_fn(prompt: str, gen_kwargs: Dict[str, Any]) -> GenerationStats:
        del gen_kwargs  # the constructor's decoding settings apply to every request
        start = time.time()
        output, tokens, calls, drafts = generator.generate_with_draft_speculation(prompt=prompt)
        text = output[0] if isinstance(output, list) else str(output)
        return GenerationStats(text, int(tokens), int(calls), int(drafts), time.time() - start)

    return generate_fn


def run_method(method: str, label: str, generator: Any, tokenizer: Any, model_key: str,
               dataset_name: str, device: torch.device, args: argparse.Namespace,
               calibration_path: Optional[str]) -> Dict[str, Any]:
    """Run lm-eval (and EvalPlus for code) with one generator; return the result record."""
    logger.info("=" * 60)
    logger.info("Evaluating %s on %s (%s)", model_key, dataset_name, label)
    logger.info("=" * 60)
    set_seed(args.seed)
    ensure_lmeval_runtime_env()
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager

    if dataset_name in {"minervamath", "math500"}:
        patch_minerva_math_process_results()
    lm_eval_config = get_lmeval_dataset_config(dataset_name)
    num_fewshot = lm_eval_config.num_fewshot if args.num_fewshot is None else args.num_fewshot

    is_nld = model_key in NLD_MODELS
    # Fast-dLLM v2's official eval rewrites math prompts to ask for a boxed answer;
    # NLD answers in LaTeX, so its boxed value is surfaced for the gsm8k filters.
    prompt_preprocessor = (
        fastdllm_v2_official_math_prompt_rewrite
        if dataset_name in {"gsm8k", "minervamath", "math500"} and not is_nld else None
    )
    output_postprocessor = boxed_answer_postprocessor if is_nld and dataset_name == "gsm8k" else None

    lm = DLLMEvalAdapter(
        generate_fn=make_generate_fn(generator),
        tokenizer_name=f"{model_key}:{label}",
        tokenizer=tokenizer,
        trim_until=dataset_name not in EVALPLUS_DATASETS,
        prompt_preprocessor=prompt_preprocessor,
        output_postprocessor=output_postprocessor,
    )
    lm_eval_result = evaluator.simple_evaluate(
        model=lm,
        tasks=list(lm_eval_config.tasks),
        num_fewshot=num_fewshot,
        limit=args.num_samples,
        log_samples=True,
        task_manager=TaskManager(),
        bootstrap_iters=0,
        confirm_run_unsafe_code=True,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    correctness = primary_metric_from_results(lm_eval_result, lm_eval_config)
    totals = lm.totals()
    fold_ratio = getattr(generator, "fold_ratio", None)
    if fold_ratio is not None and fold_ratio == fold_ratio:  # not NaN
        totals["fold_ratio"] = round(float(fold_ratio), 4)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evalplus_dir = os.path.join(args.output_dir, model_key, dataset_name,
                                f"{model_key}__{dataset_name}__{label}_{timestamp}_evalplus")
    evalplus_summary = run_evalplus_scoring(
        lm_eval_result=lm_eval_result, lm_eval_config=lm_eval_config, dataset_name=dataset_name,
        evalplus_dir=evalplus_dir, model_key=model_key,
    )
    if evalplus_summary is not None and evalplus_summary.get("status") == "completed":
        correctness["accuracy"] = evalplus_summary.get(EVALPLUS_ACCURACY_KEY.get(dataset_name, "plus"))
        correctness["source"] = "evalplus"
    elif evalplus_summary is not None:
        logger.warning("[%s] EvalPlus scoring failed: %s", label, evalplus_summary.get("error"))

    protocol = {
        "framework": "lm_eval",
        **lm_eval_config.to_dict(),
        "requested_num_fewshot": num_fewshot,
        "apply_chat_template": args.apply_chat_template,
        "fewshot_as_multiturn": args.fewshot_as_multiturn,
        "max_new_tokens": args.max_new_tokens,
        "block_length": args.block_length,
        "small_block_size": args.small_block_size,
        "threshold": args.threshold,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "use_block_cache": args.use_block_cache,
        "fastdllm_official_math_prompt_rewrite": prompt_preprocessor is not None,
        "boxed_answer_postprocessor": output_postprocessor is not None,
    }
    if method != "vanilla":
        protocol.update(
            budget_B=args.budget_B, depth_D=args.depth_D, per_step_limit=args.per_step_limit,
            calibration_set=args.calibration_set,
            calibration_num_samples=min(Config.NUM_SAMPLES, calibration_set_size(args.calibration_set)),
        )
    if method == "specfold":
        protocol.update(delta=args.delta)
    spec = get_model_spec(model_key)
    result = {
        "status": "completed",
        "model_key": model_key,
        "model_display_name": spec.display_name,
        "pretrained_path": args.pretrained_path or spec.pretrained_path,
        "dataset_name": dataset_name,
        "strategy": label,
        "num_requested": args.num_samples,
        "num_completed": len(lm.records),
        "gpu_name": current_gpu_name(device),
        "calibration_path": calibration_path if method != "vanilla" else None,
        "evaluation_protocol": protocol,
        "correctness": correctness,
        "totals": totals,
        "lm_eval": compact_lmeval_result(lm_eval_result),
        "evalplus": evalplus_summary,
        "samples": lm.records,
    }
    return result


def save_json(obj: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def summarize(result: Dict[str, Any]) -> Dict[str, Any]:
    """NFE (model calls per sample), TPS (tokens per second), accuracy, fold ratio."""
    t = result["totals"]
    n = max(result["num_completed"], 1)
    return {
        "nfe": t["model_calls"] / n,
        "tps": t["tokens_generated"] / t["elapsed_s"] if t.get("elapsed_s") else None,
        "accuracy": result["correctness"].get("accuracy"),
        "fold_ratio": t.get("fold_ratio"),
        "num_completed": result["num_completed"],
        "tokens_generated": t["tokens_generated"],
        "elapsed_s": t["elapsed_s"],
    }


def print_table(summary: Dict[str, Any]) -> None:
    rows = summary["methods"]
    vanilla_tps = rows.get("vanilla", {}).get("tps")
    s = summary["settings"]
    print(f"\n{summary['model']} / {summary['dataset']}  "
          f"(B={s['budget_B']}, D={s['depth_D']}, S={s['per_step_limit']}, delta={s['delta']}, "
          f"threshold={s['threshold']}, {summary['gpu_name']})")
    print(f"{'method':<10}{'NFE':>8}{'TPS':>9}{'gain':>8}{'acc':>9}{'fold':>7}")
    for method, r in rows.items():
        gain = f"{r['tps'] / vanilla_tps:.2f}x" if vanilla_tps and r["tps"] else "-"
        acc = f"{100 * r['accuracy']:.2f}%" if isinstance(r["accuracy"], (int, float)) else "-"
        fold = f"{r['fold_ratio']:.2f}" if r.get("fold_ratio") is not None else ""
        print(f"{method:<10}{r['nfe']:>8.1f}{r['tps']:>9.1f}{gain:>8}{acc:>9}{fold:>7}")
    print()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_key, dataset_name = args.model, args.dataset
    out_dir = os.path.join(args.output_dir, model_key, dataset_name)

    graph, calibration_path = None, None
    if any(m != "vanilla" for m in args.methods):
        graph, calibration_path = load_or_build_calibration_graph(
            model_key=model_key,
            pretrained_path=args.pretrained_path,
            device=args.device,
            dtype=args.dtype,
            block_length=args.block_length,
            temperature=args.temperature,
            depth_D=args.depth_D,
            budget_B=args.budget_B,
            per_step_limit=args.per_step_limit,
            calibration_suffix=args.calibration_suffix,
            calibration_set=args.calibration_set,
        )

    executor = load_executor(model_key, device, dtype, args.pretrained_path)
    tokenizer = executor.get_tokenizer()

    summary: Dict[str, Any] = {
        "model": model_key,
        "dataset": dataset_name,
        "gpu_name": current_gpu_name(device),
        "num_samples": args.num_samples,
        "settings": {
            "budget_B": args.budget_B, "depth_D": args.depth_D, "per_step_limit": args.per_step_limit,
            "calibration_set": args.calibration_set, "calibration_path": calibration_path,
            "delta": args.delta, "threshold": args.threshold,
            "block_length": args.block_length, "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature, "seed": args.seed,
        },
        "methods": {},
        "result_files": {},
    }
    for method in args.methods:
        label = result_label(method, args)
        generator = build_generator(method, model_key, executor, graph, device, args)
        try:
            result = run_method(method, label, generator, tokenizer, model_key, dataset_name, device, args,
                                calibration_path)
        except Exception:
            logger.exception("Failed: %s", label)
            result = {"status": "failed", "model_key": model_key, "dataset_name": dataset_name, "strategy": label}
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = save_json(result, os.path.join(out_dir, f"{model_key}__{dataset_name}__{label}_{stamp}.json"))
        summary["result_files"][method] = path
        if result["status"] == "completed":
            summary["methods"][method] = summarize(result)
        logger.info("Saved %s", path)
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vanilla_tps = summary["methods"].get("vanilla", {}).get("tps")
    summary["tps_gain_over_vanilla"] = {
        m: r["tps"] / vanilla_tps for m, r in summary["methods"].items() if vanilla_tps and r["tps"]
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = save_json(summary, os.path.join(out_dir, f"summary_{stamp}.json"))
    print_table(summary)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
