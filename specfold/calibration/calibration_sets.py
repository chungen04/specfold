"""Calibration prompt sets (Spiffy protocol): ``spiffy_mix`` = 25 MBPP-train prompts
(excluding the task ids that MBPP+ evaluates) + 25 MATH-train prompts (stratified over
the seven subjects), rendered by the same lm-eval tasks as the evaluation (0-shot, chat
template) and verified disjoint from every evaluated document."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from lm_eval.tasks import TaskManager, get_task_dict

from specfold.evaluate.lmeval_adapter import (
    LMEvalPromptSample,
    _flatten_task_dict,
    fastdllm_v2_official_math_prompt_rewrite,
)

logger = logging.getLogger(__name__)

MATH_SUBJECT_TASKS = (
    "minerva_math_algebra",
    "minerva_math_counting_and_prob",
    "minerva_math_geometry",
    "minerva_math_intermediate_algebra",
    "minerva_math_num_theory",
    "minerva_math_prealgebra",
    "minerva_math_precalc",
)

# name -> ((source, num_prompts), ...)
CALIBRATION_SETS: Dict[str, Tuple[Tuple[str, int], ...]] = {
    "spiffy_mix": (("mbpp_train", 25), ("math_train", 25)),
}


def calibration_set_size(name: str) -> int:
    return sum(n for _, n in CALIBRATION_SETS[name])


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _mbppplus_task_ids() -> set:
    from datasets import load_dataset
    return set(int(t) for t in load_dataset("evalplus/mbppplus", split="test")["task_id"])


def _stratify(total: int, groups: int) -> List[int]:
    base, extra = divmod(total, groups)
    return [base + (1 if i < extra else 0) for i in range(groups)]


def _instances_for_task(task_obj, split: str, limit: Optional[int], chat_template, tokenizer_name: str):
    available = getattr(task_obj, "dataset", None)
    if available is None or split not in available:
        raise RuntimeError(
            f"calibration split '{split}' is not loaded for task {task_obj.config.task!r} "
            f"(available: {list(available.keys()) if available is not None else None}); refusing to fall back"
        )
    task_obj.set_config(key="test_split", value=split)
    task_obj.set_config(key="num_fewshot", value=0)
    task_obj.set_fewshot_seed(seed=42)
    task_obj.build_all_requests(
        limit=limit,
        rank=0,
        world_size=1,
        cache_requests=False,
        system_instruction=None,
        apply_chat_template=chat_template is not None,
        fewshot_as_multiturn=True,
        chat_template=chat_template,
        tokenizer_name=tokenizer_name,
    )
    return [i for i in task_obj.instances if i.request_type == "generate_until"]


def load_calibration_set_prompts(
    name: str,
    model_key: str,
    chat_template: Optional[Callable[..., str]] = None,
    tokenizer_name: str = "specfold",
) -> Tuple[List[LMEvalPromptSample], Dict[str, Any]]:
    """Return (prompts, sources) for a named calibration set.

    ``sources`` maps each source to the list of document keys it contributed
    (``("mbpp", task_id)`` / ``("math", subject, train_index)``) plus the
    normalized problem text, for the disjointness check and the graph metadata.
    """
    if name not in CALIBRATION_SETS:
        raise ValueError(f"unknown calibration set {name!r}; known: {sorted(CALIBRATION_SETS)}")
    task_manager = TaskManager()
    rewrite = (
        fastdllm_v2_official_math_prompt_rewrite
        if str(model_key).startswith("fastdllm_v2") else (lambda prompt, task: prompt)
    )
    prompts: List[LMEvalPromptSample] = []
    sources: Dict[str, List[Dict[str, Any]]] = {}

    for source, n in CALIBRATION_SETS[name]:
        docs: List[Dict[str, Any]] = []
        if source == "mbpp_train":
            excluded = _mbppplus_task_ids()
            (_, task_obj), = list(_flatten_task_dict(get_task_dict(["mbpp"], task_manager)))
            kept = 0
            for inst in _instances_for_task(task_obj, "train", None, chat_template, tokenizer_name):
                tid = int(inst.doc["task_id"])
                if tid in excluded:
                    continue
                context, gen_kwargs = inst.args
                prompts.append(LMEvalPromptSample(
                    prompt=rewrite(context, "mbpp"), sample_id=f"mbpp_train:{tid}",
                    task_name="mbpp", doc_id=tid, gen_kwargs=dict(gen_kwargs or {})))
                docs.append({"key": ["mbpp", tid], "text": _norm(inst.doc["text"])})
                kept += 1
                if kept == n:
                    break
            if kept < n:
                raise RuntimeError(f"mbpp_train: only {kept} of {n} prompts available after excluding MBPP+ ids")
        elif source == "math_train":
            per_subject = _stratify(n, len(MATH_SUBJECT_TASKS))
            task_dict = get_task_dict(list(MATH_SUBJECT_TASKS), task_manager)
            tasks = dict(_flatten_task_dict(task_dict))
            for subject, k in zip(MATH_SUBJECT_TASKS, per_subject):
                insts = _instances_for_task(tasks[subject], "train", k, chat_template, tokenizer_name)
                if len(insts) < k:
                    raise RuntimeError(f"{subject}: only {len(insts)} of {k} train prompts")
                for idx, inst in enumerate(insts):
                    context, gen_kwargs = inst.args
                    prompts.append(LMEvalPromptSample(
                        prompt=rewrite(context, subject), sample_id=f"math_train:{subject}:{idx}",
                        task_name=subject, doc_id=idx, gen_kwargs=dict(gen_kwargs or {})))
                    docs.append({"key": ["math", subject, idx], "text": _norm(inst.doc["problem"])})
        else:
            raise ValueError(f"unknown calibration source {source!r}")
        sources[source] = docs
        logger.info("calibration set %s: %d prompts from %s", name, len(docs), source)
    return prompts, sources


def evaluation_document_index() -> Dict[str, Any]:
    """Identities of every document the paper's five benchmarks evaluate."""
    from datasets import load_dataset
    texts: Dict[str, set] = {}
    texts["gsm8k"] = {_norm(q) for q in load_dataset("openai/gsm8k", "main", split="test")["question"]}
    texts["math500"] = {_norm(p) for p in load_dataset("HuggingFaceH4/MATH-500", split="test")["problem"]}
    texts["humaneval"] = {_norm(p) for p in load_dataset("openai/openai_humaneval", split="test")["prompt"]}
    mbpp_plus = load_dataset("evalplus/mbppplus", split="test")
    texts["mbpp"] = {_norm(p) for p in mbpp_plus["prompt"]}
    texts["ifeval"] = {_norm(p) for p in load_dataset("google/IFEval", split="train")["prompt"]}
    return {"texts": texts, "mbpp_task_ids": set(int(t) for t in mbpp_plus["task_id"])}


def verify_disjoint_from_evaluation(sources: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Raise if any calibration document is an evaluated document; return a report."""
    index = evaluation_document_index()
    overlaps = []
    n_docs = 0
    for source, docs in sources.items():
        for d in docs:
            n_docs += 1
            if d["key"][0] == "mbpp" and d["key"][1] in index["mbpp_task_ids"]:
                overlaps.append((source, d["key"], "mbpp+ task id"))
            for bench, pool in index["texts"].items():
                if d["text"] in pool:
                    overlaps.append((source, d["key"], f"{bench} text"))
    if overlaps:
        raise RuntimeError(f"calibration documents overlap evaluation data: {overlaps[:5]} (+{len(overlaps) - 5} more)")
    report = {
        "checked_documents": n_docs,
        "evaluation_sets": {k: len(v) for k, v in index["texts"].items()},
        "overlap": 0,
    }
    logger.info("calibration/evaluation disjointness verified: %s", report)
    return report
