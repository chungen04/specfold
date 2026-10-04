"""EvalPlus scoring for humaneval/humaneval_plus/mbpp/mbpp_plus datasets.

After lm-eval generates completions, this module converts the samples to
EvalPlus format and runs `python -m evalplus.evaluate` to get base/plus pass@1.

Code extraction:
- HumanEval uses `evalplus.sanitize.sanitize()` — the official tree-sitter
  extractor that pulls the function matching `entry_point` out of arbitrary
  model output.
- MBPP uses the spiffy-reproduction stop-token postprocess
  (`_postprocess_mbpp_completion` below), which matches that repo's MBPP
  postprocessing instead of routing MBPP through the AST sanitize step.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# MBPP post-processing ports the spiffy-reproduction `postprocess_mbpp_completion`
# stop-token logic (see spiffy-reproduction/.../task_metrics.py).
_MBPP_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_MBPP_STOP_TOKENS = ("[DONE]", "[BEGIN]", "\nassert", '\n"""', "\nprint", "\nif", "\n```")
_MBPP_CODE_HEAD_RE = re.compile(r"\s*(?:def\s|import\s|from\s+\S+\s+import|class\s|@\w)")


def _postprocess_mbpp_completion(text: str) -> str:
    """spiffy-reproduction-style MBPP postprocess (no AST sanitize).

    1. If the response is wrapped in a ```python ... ``` fence, take the first
       block's content. Skip this step when the response already starts with
       code-like tokens — Fast-dLLM v2 1.5B emits the real function first and
       then puts demo asserts inside a trailing fence; grabbing the first fence
       would discard the actual code.
    2. Drop a leading `[BEGIN]` marker if the model echoed it.
    3. Truncate at the first stop token: `[DONE]`, `[BEGIN]` (a hallucinated new
       turn after the code), `\\nassert`, `\\n\"\"\"`, `\\nprint`, `\\nif`,
       or `\\n\\`\\`\\``.
    """
    text = text or ""
    if not _MBPP_CODE_HEAD_RE.match(text):
        fenced = _MBPP_CODE_BLOCK_RE.findall(text)
        if fenced:
            text = fenced[0]
    text = text.strip()
    if text.startswith("[BEGIN]"):
        text = text[len("[BEGIN]"):].lstrip()
    stop_positions = [text.find(tok) for tok in _MBPP_STOP_TOKENS if tok and tok in text]
    if stop_positions:
        text = text[: min(stop_positions)]
    return text.rstrip()

# Maps our dataset_name → evalplus dataset identifier.
# All four variants route through EvalPlus because instruction-tuned outputs
# (Fast-dLLM v2 chat-template) wrap code in markdown fences that EvalPlus's
# sanitize strips but lm-eval's raw pass_at_1 does not. The plain/plus
# variants share the same generations and differ only in which subscore is
# reported as the primary metric.
#
# Note: plain mbpp must use lm-eval task `mbpp_plus` (not `mbpp`). lm-eval's
# `mbpp` task runs the full MBPP test split (500 problems, task IDs 11-510),
# which only overlaps EvalPlus's mbpp_plus 378-problem subset on 224 IDs —
# scoring those generations via EvalPlus silently drops half the data. The
# `mbpp_plus` task generates samples on EvalPlus's exact 378-task-ID set.
EVALPLUS_DATASETS: Dict[str, str] = {
    "humaneval": "humaneval",
    "humaneval_plus": "humaneval",
    "mbpp": "mbpp",
    "mbpp_plus": "mbpp",
}

# For plain variants report base pass@1; for *_plus variants report plus pass@1.
EVALPLUS_ACCURACY_KEY: Dict[str, str] = {
    "humaneval": "base",
    "humaneval_plus": "plus",
    "mbpp": "base",
    "mbpp_plus": "plus",
}

_DATASET_PREFIX = {"mbpp": "Mbpp", "humaneval": "HumanEval"}


def _load_evalplus_problems(dataset: str) -> Dict[str, Dict[str, Any]]:
    """Load the full EvalPlus problem dict for `dataset` (task_id → problem)."""
    try:
        if dataset == "mbpp":
            from evalplus.data import get_mbpp_plus  # noqa: WPS433
            return get_mbpp_plus()
        if dataset == "humaneval":
            from evalplus.data import get_human_eval_plus  # noqa: WPS433
            return get_human_eval_plus()
    except Exception as exc:  # pragma: no cover - import-time guard
        raise RuntimeError(
            "EvalPlus scoring requires the evalplus package. "
            "Install with: pip install pydantic==2.10.6 evalplus==0.3.1"
        ) from exc
    raise ValueError(f"Unsupported EvalPlus dataset: {dataset!r}")


def _make_task_id(doc: dict, dataset: str) -> str:
    raw_id = doc["task_id"]
    prefix = _DATASET_PREFIX[dataset]
    if isinstance(raw_id, str) and raw_id.startswith(prefix):
        return raw_id
    return f"{prefix}/{raw_id}"


def _samples_for_task(
    lm_eval_result: Dict[str, Any],
    task_names: Tuple[str, ...],
) -> Tuple[str, list]:
    samples_by_task = lm_eval_result.get("samples") or {}
    for task_name in task_names:
        samples = samples_by_task.get(task_name)
        if samples:
            return task_name, list(samples)
    if len(samples_by_task) == 1:
        task_name, samples = next(iter(samples_by_task.items()))
        if samples:
            return task_name, list(samples)
    raise RuntimeError(f"Could not find lm-eval samples for tasks {list(task_names)}.")


def _sanitize_response(
    response: str,
    problem: Dict[str, Any],
    evalplus_dataset: str,
) -> str:
    """Extract the candidate code for one sample.

    - HumanEval: prepend the EvalPlus prompt (signature + docstring) and run it
      through `evalplus.sanitize.sanitize()`, matching the canonical EvalPlus
      ``completion`` flow.
    - MBPP: use the spiffy-reproduction stop-token postprocess. MBPP outputs are
      typically a full `def ...` block, sometimes followed by demo asserts /
      prose / hallucinated `[BEGIN]` markers; the stop-token approach handles
      those without forcing the AST extractor to find a name match.
    """
    if evalplus_dataset == "humaneval":
        from evalplus.sanitize import sanitize  # noqa: WPS433
        # Fast-dLLM v2 chat outputs wrap the solution in a ```python ... ``` fence
        # followed by prose. Passing that raw to sanitize() yields invalid Python
        # (the fence markers + prose break the AST parse), so sanitize falls back
        # to garbage (e.g. just the import line). Extract the fenced code first.
        resp = response or ""
        fenced = _MBPP_CODE_BLOCK_RE.findall(resp)
        if fenced:
            resp = fenced[0]
        code = problem.get("prompt", "") + "\n" + resp
        return sanitize(code=code, entrypoint=problem.get("entry_point"))
    return _postprocess_mbpp_completion(response)


def run_evalplus_scoring(
    lm_eval_result: Dict[str, Any],
    lm_eval_config: Any,
    dataset_name: str,
    evalplus_dir: str,
    model_key: str,
    strategy: str = "vanilla",
) -> Optional[Dict[str, Any]]:
    """Run EvalPlus scoring on lm-eval samples. Returns None if not an EvalPlus dataset."""
    evalplus_dataset = EVALPLUS_DATASETS.get(dataset_name)
    if evalplus_dataset is None:
        return None

    start = time.time()
    problems = _load_evalplus_problems(evalplus_dataset)
    all_task_ids: Set[str] = set(problems.keys())

    os.makedirs(evalplus_dir, exist_ok=True)
    task_name, samples = _samples_for_task(lm_eval_result, lm_eval_config.tasks)

    sample_path = os.path.join(evalplus_dir, f"samples_{task_name}_{model_key}__{strategy}.jsonl")
    with open(sample_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")

    evalplus_path = os.path.join(evalplus_dir, f"evalplus_{evalplus_dataset}_samples.jsonl")
    generated_ids: Set[str] = set()
    with open(evalplus_path, "w") as f:
        for sample in samples:
            task_id = _make_task_id(sample["doc"], evalplus_dataset)
            generated_ids.add(task_id)
            resp = sample["filtered_resps"][0]
            if isinstance(resp, list):
                resp = resp[0]
            problem = problems.get(task_id)
            if problem is None:
                # Generated for a task ID not in EvalPlus; will be ignored by
                # the scoring loop below. Pass the raw response through.
                solution = resp
            else:
                solution = _sanitize_response(resp, problem, evalplus_dataset)
            f.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")
        for task_id in all_task_ids - generated_ids:
            f.write(json.dumps({"task_id": task_id, "solution": "pass"}) + "\n")

    results_path = evalplus_path.replace(".jsonl", "_eval_results.json")
    if os.path.exists(results_path):
        os.remove(results_path)

    proc = subprocess.run(
        [sys.executable, "-m", "evalplus.evaluate", "--dataset", evalplus_dataset,
         "--samples", evalplus_path, "--i-just-wanna-run"],
        capture_output=True, text=True,
    )
    if not os.path.exists(results_path):
        error_msg = proc.stderr[-400:] if proc.stderr else "no output"
        logger.error("EvalPlus failed for %s: %s", dataset_name, error_msg)
        return {
            "status": "failed",
            "dataset": evalplus_dataset,
            "error": f"evalplus failed: {error_msg}",
            "elapsed_s": time.time() - start,
        }

    with open(results_path) as f:
        data = json.load(f)

    base_pass = plus_pass = total = 0
    for task_id, res in data.get("eval", {}).items():
        if task_id not in generated_ids:
            continue
        total += 1
        if res[0]["base_status"] == "pass":
            base_pass += 1
        if res[0]["plus_status"] == "pass":
            plus_pass += 1

    return {
        "status": "completed",
        "dataset": evalplus_dataset,
        "lm_eval_task": task_name,
        "lm_eval_sample_count": len(samples),
        "total": total,
        "base": base_pass / total if total else 0.0,
        "plus": plus_pass / total if total else 0.0,
        "avg": (base_pass + plus_pass) / (2 * total) if total else 0.0,
        "lm_eval_samples_path": sample_path,
        "evalplus_samples_path": evalplus_path,
        "evalplus_results_path": results_path,
        "elapsed_s": time.time() - start,
    }
