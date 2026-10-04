"""lm-eval integration for Spiffy generation strategies."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple



logger = logging.getLogger(__name__)

PromptPreprocessor = Callable[[str, Optional[str]], str]
_MINERVA_PROCESS_RESULTS_PATCHED = False


def ensure_lmeval_runtime_env(cache_root: Optional[str] = None) -> str:
    """Set writable runtime locations before lm-eval/evaluate import task code."""
    resolved_root = os.path.abspath(cache_root or os.environ.get("SPIFFY_LMEVAL_CACHE", "cache/lm_eval_hf"))
    os.makedirs(resolved_root, exist_ok=True)

    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    hf_home = os.path.abspath(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))

    def can_write(path: str) -> bool:
        probe = path
        while not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        return os.access(probe, os.W_OK)

    def ensure_cache_var(env_name: str, default_subdir: str) -> None:
        if env_name not in os.environ:
            default_path = os.path.join(hf_home, default_subdir)
            if not can_write(default_path):
                os.environ[env_name] = os.path.join(resolved_root, default_subdir)
        if env_name in os.environ:
            os.makedirs(os.environ[env_name], exist_ok=True)

    ensure_cache_var("HF_DATASETS_CACHE", "datasets")
    ensure_cache_var("HF_METRICS_CACHE", "metrics")
    ensure_cache_var("HF_EVALUATE_CACHE", "evaluate")
    return resolved_root


ensure_lmeval_runtime_env()

from lm_eval.api.model import LM  # noqa: E402


@dataclass(frozen=True)
class LMEvalDatasetConfig:
    """Dataset-level policy for the lm-eval task used by this codebase."""

    dataset_name: str
    tasks: Tuple[str, ...]
    num_fewshot: int
    primary_task: str
    primary_metric: str
    accuracy_label: str
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


LMEVAL_DATASETS: Dict[str, LMEvalDatasetConfig] = {
    "gsm8k": LMEvalDatasetConfig(
        dataset_name="gsm8k",
        tasks=("gsm8k",),
        num_fewshot=0,
        primary_task="gsm8k",
        primary_metric="exact_match,flexible-extract",
        accuracy_label="gsm8k exact_match flexible-extract",
        notes="Uses lm-eval GSM8K with the Fast-dLLM v2 official 0-shot setting.",
    ),
    "humaneval": LMEvalDatasetConfig(
        dataset_name="humaneval",
        tasks=("humaneval",),
        num_fewshot=0,
        primary_task="humaneval",
        primary_metric="pass@1,create_test",
        accuracy_label="HumanEval pass@1",
        notes="Uses lm-eval HumanEval code_eval task with unsafe code execution explicitly enabled.",
    ),
    "humaneval_plus": LMEvalDatasetConfig(
        dataset_name="humaneval_plus",
        tasks=("humaneval_plus",),
        num_fewshot=0,
        primary_task="humaneval_plus",
        primary_metric="pass@1,create_test",
        accuracy_label="HumanEval+ pass@1",
        notes="Uses lm-eval HumanEval+ EvalPlus task with unsafe code execution explicitly enabled.",
    ),
    "mbpp": LMEvalDatasetConfig(
        dataset_name="mbpp",
        tasks=("mbpp_plus",),
        num_fewshot=0,
        primary_task="mbpp_plus",
        primary_metric="pass_at_1,none",
        accuracy_label="MBPP pass@1",
        notes=(
            "Uses lm-eval mbpp_plus task (EvalPlus's 378-problem set) at 0-shot "
            "to match the Fast-dLLM v2 paper setup; final accuracy comes from "
            "EvalPlus base scoring."
        ),
    ),
    "mbpp_plus": LMEvalDatasetConfig(
        dataset_name="mbpp_plus",
        tasks=("mbpp_plus",),
        num_fewshot=0,
        primary_task="mbpp_plus",
        primary_metric="pass_at_1,none",
        accuracy_label="MBPP+ pass@1",
        notes=(
            "Uses lm-eval mbpp_plus task (EvalPlus's 378-problem set) at 0-shot "
            "to match the Fast-dLLM v2 paper setup; final accuracy comes from "
            "EvalPlus plus scoring."
        ),
    ),
    "minervamath": LMEvalDatasetConfig(
        dataset_name="minervamath",
        tasks=("minerva_math",),
        num_fewshot=0,
        primary_task="minerva_math",
        primary_metric="math_verify,none",
        accuracy_label="Minerva Math math_verify",
        notes="Uses lm-eval minerva_math aggregate task with an explicit 0-shot override.",
    ),
    "math500": LMEvalDatasetConfig(
        dataset_name="math500",
        tasks=("minerva_math500",),
        num_fewshot=0,
        primary_task="minerva_math500",
        primary_metric="math_verify,none",
        accuracy_label="MATH-500 math_verify",
        notes=(
            "Uses lm-eval minerva_math500 (HuggingFaceH4/MATH-500, 500 problems) "
            "with an explicit 0-shot override; shares minerva_math prompt/scoring "
            "code, so patch_minerva_math_process_results applies."
        ),
    ),
    "ifeval": LMEvalDatasetConfig(
        dataset_name="ifeval",
        tasks=("ifeval",),
        num_fewshot=0,
        primary_task="ifeval",
        primary_metric="prompt_level_strict_acc,none",
        accuracy_label="IFEval prompt-level strict accuracy",
        notes="Uses lm-eval IFEval strict/loose instruction-following metrics.",
    ),
}


@dataclass
class LMEvalPromptSample:
    """A calibration prompt generated by lm-eval request construction."""

    prompt: str
    sample_id: str
    task_name: str
    doc_id: Optional[int]
    gen_kwargs: Dict[str, Any]


def get_lmeval_dataset_config(dataset_name: str) -> LMEvalDatasetConfig:
    from specfold.registry import normalize_dataset_name

    normalized = normalize_dataset_name(dataset_name)
    try:
        return LMEVAL_DATASETS[normalized]
    except KeyError as exc:  # pragma: no cover - registry guards this
        raise ValueError(f"No lm-eval mapping is registered for dataset {dataset_name!r}.") from exc


def fastdllm_v2_official_math_prompt_rewrite(prompt: str, task_name: Optional[str]) -> str:
    """Match Fast-dLLM v2 eval.py math prompt preprocessing."""
    task = str(task_name or "")
    if task.startswith("minerva_math"):
        return prompt.replace(
            "Solution:",
            "Please reason step by step, and put your final answer within \\boxed{{}}.",
        )
    if task.startswith("gsm8k"):
        return prompt.replace(
            "Answer:",
            "Please reason step by step, and put your final answer within \\boxed{{}}.",
        )
    return prompt


def boxed_answer_postprocessor(text: str) -> str:
    """Append a plain-text final answer extracted from the last \\boxed{...}.

    Nemotron-Labs-Diffusion instruct models answer math questions in LaTeX
    display blocks (``$$ \\boxed{18} $$``). lm-eval's gsm8k flexible-extract
    filter takes the LAST regex match of ``(-?[$0-9.,]{2,})|(-?[0-9]+)``, which
    matches the trailing ``$$`` delimiter instead of the answer. Appending
    "The answer is <x>." makes both the strict and flexible gsm8k filters see
    the boxed value; outputs without \\boxed are returned unchanged.
    """
    import re

    matches = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if not matches:
        return text
    answer = matches[-1].strip().strip("$").replace(",", "").replace("\\!", "").strip()
    if not answer:
        return text
    return f"{text}\nThe answer is {answer}."


def patch_minerva_math_process_results() -> bool:
    """Patch lm-eval Minerva scoring to match the Fast-dLLM v2 eval.py behavior."""
    global _MINERVA_PROCESS_RESULTS_PATCHED
    if _MINERVA_PROCESS_RESULTS_PATCHED:
        return True

    try:
        from math_verify import parse, verify
        import lm_eval.tasks.minerva_math.utils as minerva_utils
    except ImportError as exc:
        logger.warning("Could not patch minerva_math process_results: %s", exc)
        return False

    def _patched_process_results(doc: dict, results: list[str]) -> dict[str, int]:
        candidates = results[0]

        try:
            parsed_gold = parse(doc["answer"])
        except Exception:
            parsed_gold = []
        if not parsed_gold:
            try:
                parsed_gold = parse(r"\boxed{" + doc["answer"] + "}")
            except Exception:
                parsed_gold = []

        try:
            mv_res = verify(parsed_gold, parse(candidates))
        except Exception:
            mv_res = False

        eq_res = False
        try:
            boxed_str = minerva_utils.last_boxed_only_string(candidates)
            if boxed_str:
                extracted = minerva_utils.normalize_final_answer(
                    minerva_utils.remove_boxed(boxed_str)
                )
                eq_res = minerva_utils.is_equiv(extracted, doc["answer"])
        except Exception:
            pass

        return {
            "exact_match": 1 if eq_res else 0,
            "math_verify": 1 if (mv_res or eq_res) else 0,
        }

    minerva_utils.process_results = _patched_process_results
    _MINERVA_PROCESS_RESULTS_PATCHED = True
    logger.info("Patched lm_eval.tasks.minerva_math.utils.process_results for Fast-dLLM v2 compatibility.")
    return True


def trim_at_stop_sequences(text: str, until: Any) -> str:
    """Apply lm-eval generate_until stop strings to a model continuation."""
    if until is None:
        return text
    stops = [until] if isinstance(until, str) else list(until)
    stops = [stop for stop in stops if isinstance(stop, str) and stop]
    if not stops:
        return text

    earliest: Optional[int] = None
    for stop in stops:
        pos = text.find(stop)
        if pos >= 0 and (earliest is None or pos < earliest):
            earliest = pos
    return text if earliest is None else text[:earliest]


class DLLMEvalAdapter(LM):
    """lm-eval LM wrapper around one Spiffy generation callable."""

    def __init__(
        self,
        generate_fn: Callable[[str, Dict[str, Any]], Any],
        tokenizer_name: str = "spiffy",
        tokenizer: Optional[Any] = None,
        trim_until: bool = True,
        prompt_preprocessor: Optional[PromptPreprocessor] = None,
        output_postprocessor: Optional[Callable[[str], str]] = None,
    ) -> None:
        super().__init__()
        self._generate_fn = generate_fn
        self._tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.trim_until = trim_until
        self.prompt_preprocessor = prompt_preprocessor
        self.output_postprocessor = output_postprocessor
        self.records: List[Dict[str, Any]] = []

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        if self.tokenizer is None or not hasattr(self.tokenizer, "apply_chat_template"):
            raise NotImplementedError(
                "DLLMEvalAdapter needs a tokenizer with apply_chat_template() when lm-eval chat formatting is enabled."
            )
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        raise NotImplementedError("DLLMEvalAdapter only supports lm-eval generate_until tasks.")

    def loglikelihood_rolling(self, requests) -> list[float]:
        raise NotImplementedError("DLLMEvalAdapter only supports lm-eval generate_until tasks.")

    def generate_until(self, requests) -> list[str]:
        outputs: List[str] = []
        for request_index, request in enumerate(requests):
            context, gen_kwargs = request.args
            gen_kwargs = dict(gen_kwargs or {})
            task_name = getattr(request, "task_name", None)
            runtime_context = (
                self.prompt_preprocessor(context, task_name)
                if self.prompt_preprocessor is not None
                else context
            )
            stats = self._generate_fn(runtime_context, gen_kwargs)
            raw_output = str(getattr(stats, "output", ""))
            processed = self.output_postprocessor(raw_output) if self.output_postprocessor else raw_output
            output = trim_at_stop_sequences(processed, gen_kwargs.get("until")) if self.trim_until else processed
            prompt_rewritten = runtime_context != context
            record = {
                "request_index": request_index,
                "task_name": task_name,
                "doc_id": getattr(request, "doc_id", None),
                "prompt": runtime_context,
                "prompt_rewritten": prompt_rewritten,
                "generation_kwargs": gen_kwargs,
                "output": output,
                "raw_output": raw_output,
                "tokens_generated": int(getattr(stats, "tokens_generated", 0)),
                "model_calls": int(getattr(stats, "model_calls", 0)),
                "drafts_computed": int(getattr(stats, "drafts_computed", 0)),
                "elapsed_s": float(getattr(stats, "elapsed_s", 0.0)),
                "step_latency_s": (
                    float(getattr(stats, "elapsed_s", 0.0))
                    / max(1, int(getattr(stats, "model_calls", 1)))
                ),
            }
            if prompt_rewritten:
                record["original_prompt"] = context
            self.records.append(record)
            outputs.append(output)
        return outputs

    def totals(self) -> Dict[str, Any]:
        n = len(self.records)
        total_elapsed = sum(r.get("elapsed_s", 0.0) for r in self.records)
        total_calls = sum(r.get("model_calls", 0) for r in self.records)
        return {
            "tokens_generated": sum(r.get("tokens_generated", 0) for r in self.records),
            "model_calls": total_calls,
            "drafts_computed": sum(r.get("drafts_computed", 0) for r in self.records),
            "elapsed_s": total_elapsed,
            "avg_elapsed_s": total_elapsed / n if n > 0 else 0.0,
            "avg_step_latency_s": total_elapsed / total_calls if total_calls > 0 else 0.0,
        }


def _flatten_task_dict(task_dict: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    for task_name, task_obj in task_dict.items():
        if isinstance(task_obj, dict):
            yield from _flatten_task_dict(task_obj)
        else:
            yield task_name, task_obj


def primary_metric_from_results(
    lm_eval_result: Dict[str, Any],
    config: LMEvalDatasetConfig,
) -> Dict[str, Any]:
    """Extract the codebase's single accuracy value from lm-eval results."""
    results = lm_eval_result.get("results") or {}
    task_result = results.get(config.primary_task) or {}
    metric_value = task_result.get(config.primary_metric)

    if metric_value is None and config.dataset_name in {"humaneval", "humaneval_plus"}:
        for key, value in task_result.items():
            if key.startswith("pass@1"):
                metric_value = value
                break

    if metric_value is None and config.dataset_name == "minervamath":
        metric_values = [
            value
            for task_name, task_result in results.items()
            if task_name.startswith("minerva_math_")
            for metric_name, value in task_result.items()
            if metric_name == config.primary_metric and isinstance(value, (int, float))
        ]
        if metric_values:
            metric_value = sum(metric_values) / len(metric_values)

    if metric_value is None and task_result:
        for metric_name, value in task_result.items():
            if metric_name.endswith(",none") and isinstance(value, (int, float)):
                metric_value = value
                break

    return {
        "num_scored": None,
        "num_correct": None,
        "accuracy": metric_value,
        "source": "lm_eval",
        "task": config.primary_task,
        "metric": config.primary_metric,
        "label": config.accuracy_label,
    }


def compact_lmeval_result(lm_eval_result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep aggregate lm-eval output separate from Spiffy per-generation records."""
    return {
        key: value
        for key, value in lm_eval_result.items()
        if key not in {"samples"}
    }
