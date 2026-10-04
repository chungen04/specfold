"""Draft-graph calibration (Spiffy): record which (position-rank, vocab-rank) unmasking
patterns the model's own decoding commits, then select a budget-B graph of them.
Family pipelines subclass this and supply ``setup_model``, ``prepare_input`` and
``_collect_sample_calibration_data``."""

from __future__ import annotations

import gc
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from specfold.calibration.calibration_sets import (
    calibration_set_size,
    load_calibration_set_prompts,
    verify_disjoint_from_evaluation,
)
from specfold.calibration.graph import Node, can_be_child, count_tree_nodes
from specfold.config import Config
from specfold.registry import get_model_spec

logger = logging.getLogger(__name__)


class BaseCalibrationPipeline:
    def __init__(
        self,
        model_key: str,
        pretrained_path: Optional[str] = None,
        device: str = Config.DEVICE,
        dtype: str = "bfloat16",
        num_samples: int = Config.NUM_SAMPLES,
        max_seq_len: int = Config.MAX_SEQ_LEN,
        gen_length: int = Config.GEN_LENGTH,
        block_length: int = Config.BLOCK_LENGTH,
        depth_D: int = Config.DEPTH_D,
        budget_B: int = Config.BUDGET_B,
        per_step_limit: int = Config.PER_STEP_LIMIT,
        temperature: float = Config.TEMPERATURE,
        use_chat_template: bool = True,
        calibration_set: str = Config.CALIBRATION_SET,
    ):
        spec = get_model_spec(model_key)
        self.model_key = spec.key
        self.model_family = spec.family
        self.pretrained_path = pretrained_path or spec.pretrained_path
        self.device = device
        self.dtype = dtype
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.gen_length = gen_length
        self.block_length = block_length
        self.depth_D = min(depth_D, block_length)
        self.budget_B = budget_B
        self.per_step_limit = per_step_limit
        self.temperature = temperature
        self.use_chat_template = use_chat_template
        self.calibration_set = calibration_set
        self.calibration_sources: Optional[Dict[str, Any]] = None
        self.disjointness_check: Optional[Dict[str, Any]] = None

        self.model: Optional[Any] = None
        self.samples: List[Any] = []

    def setup_model(self) -> None:
        raise NotImplementedError("Subclasses must implement setup_model().")

    def prepare_input(
        self,
        prompt: str,
        apply_chat_template: Optional[bool] = None,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement prepare_input().")

    def _collect_sample_calibration_data(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[List[Tuple[int, Tuple[int, int]]], List[Tuple[int, List[Tuple[int, int]]]]]:
        raise NotImplementedError(
            "Subclasses must implement _collect_sample_calibration_data()."
        )

    def setup_dataset(self) -> None:
        tokenizer = getattr(getattr(self, "model", None), "tokenizer", None) or getattr(self, "tokenizer", None)
        chat_template = None
        if self.use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            def chat_template(chat_history, add_generation_prompt=True):
                return tokenizer.apply_chat_template(
                    chat_history,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
        elif self.use_chat_template:
            logger.warning("Calibration chat template requested, but no tokenizer apply_chat_template() is available.")
        self.samples, self.calibration_sources = load_calibration_set_prompts(
            self.calibration_set,
            model_key=self.model_key,
            chat_template=chat_template,
            tokenizer_name=str(self.pretrained_path or self.model_key),
        )
        self.disjointness_check = verify_disjoint_from_evaluation(self.calibration_sources)
        self.num_samples = min(self.num_samples, calibration_set_size(self.calibration_set))
        logger.info("Loaded %d calibration prompts from set %s", len(self.samples), self.calibration_set)

    def collect_calibration_data(
        self,
    ) -> Tuple[List[Tuple[int, Tuple[int, int]]], List[Tuple[int, List[Tuple[int, int]]]]]:
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        if not self.samples:
            raise RuntimeError("Dataset not initialized. Call setup_dataset() first.")

        logger.info("Collecting calibration data during model inference...")
        results: List[Tuple[int, Tuple[int, int]]] = []
        sequence_history: List[Tuple[int, List[Tuple[int, int]]]] = []
        max_samples = min(len(self.samples), self.num_samples)

        for idx, sample in enumerate(self.samples[:max_samples]):
            sid = getattr(sample, "sample_id", idx)
            logger.info("Processing calibration sample %d/%d (%s)", idx + 1, max_samples, sid)
            try:
                prompt = sample.prompt if hasattr(sample, "prompt") else str(sample)
                apply_chat_template = False
                input_ids = self.prepare_input(
                    prompt,
                    apply_chat_template=apply_chat_template,
                    max_length=self.max_seq_len,
                )
                sample_results, sample_history = self._collect_sample_calibration_data(input_ids)
                results.extend(sample_results)
                sequence_history.extend(sample_history)
                logger.info(
                    "  Collected %d tuples and %d history entries",
                    len(sample_results),
                    len(sample_history),
                )
            except Exception as exc:  # pragma: no cover - runtime robustness
                logger.exception("Failed to collect calibration data for sample %s: %s", sid, exc)
                continue
            finally:
                if idx % Config.MEMORY_CLEANUP_INTERVAL == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        logger.info("Collected %d total tuples and %d history entries", len(results), len(sequence_history))
        return results, sequence_history

    def construct_all_possibilities_graph(
        self,
        calibration_data: List[Tuple[int, Tuple[int, int]]],
        L_max: Optional[int] = None,
    ) -> Dict[int, Counter]:
        """Count observed draft tuples per lookahead distance."""
        L_max = self.depth_D if L_max is None else L_max
        frequency_counter: Dict[int, Counter] = defaultdict(Counter)
        for L, ij_pair in calibration_data:
            if 1 <= L <= L_max:
                frequency_counter[L][frozenset([ij_pair])] += 1
        return dict(frequency_counter)

    def select_optimal_subgraph(
        self,
        possibilities_result: Dict[int, Counter],
        budget_B: Optional[int] = None,
        sequence_history: Optional[List[Tuple[int, List[Tuple[int, int]]]]] = None,
    ) -> Dict[str, Any]:
        """Select a compact parent-compatible draft graph."""
        budget_B = self.budget_B if budget_B is None else budget_B
        if not sequence_history:
            return self.select_budget_limited_drafts([])

        counts_by_L: Dict[int, Counter] = defaultdict(Counter)
        for L, ij_list in sequence_history:
            if ij_list:
                counts_by_L[L][frozenset(ij_list)] += 1

        candidates_by_L: Dict[int, List[Tuple[frozenset, int]]] = {}
        for L, counts in counts_by_L.items():
            candidates_by_L[L] = counts.most_common(max(self.per_step_limit * 4, self.per_step_limit))

        if 1 not in candidates_by_L or not candidates_by_L[1]:
            return self.select_budget_limited_drafts(sequence_history)

        root_set, root_count = candidates_by_L[1][0]
        root = Node(L=1, ij_set=set(root_set), frequency=root_count, count=root_count, parent=None)
        selected_nodes: List[Node] = [root]
        visited = {root.ij_set}

        while len(selected_nodes) < budget_B:
            candidates: List[Node] = []
            for parent in selected_nodes:
                next_L = parent.L + 1
                for child_set, child_count in candidates_by_L.get(next_L, []):
                    if child_set in visited:
                        continue
                    if can_be_child(parent, next_L, set(child_set)):
                        candidates.append(
                            Node(
                                L=next_L,
                                ij_set=set(child_set),
                                frequency=child_count,
                                count=child_count,
                                parent=parent,
                            )
                        )
            if not candidates:
                fallback = self._best_unvisited_candidate(candidates_by_L, visited)
                if fallback is None:
                    break
                L, ij_set, freq = fallback
                selected_nodes.append(Node(L=L, ij_set=set(ij_set), frequency=freq, count=freq, parent=None))
                visited.add(frozenset(ij_set))
                continue

            candidates.sort(key=lambda n: n.get_score(), reverse=True)
            best = candidates[0]
            if best.parent is not None:
                best.parent.children.append(best)
            selected_nodes.append(best)
            visited.add(best.ij_set)

        return self._nodes_to_result(selected_nodes)

    def _best_unvisited_candidate(
        self,
        candidates_by_L: Dict[int, List[Tuple[frozenset, int]]],
        visited: set,
    ) -> Optional[Tuple[int, frozenset, int]]:
        best: Optional[Tuple[int, frozenset, int]] = None
        selected_counts_by_L = defaultdict(int)
        for ij_set in visited:
            selected_counts_by_L[len(ij_set)] += 1
        for L in sorted(candidates_by_L):
            if selected_counts_by_L[L] >= self.per_step_limit:
                continue
            for ij_set, freq in candidates_by_L[L]:
                if ij_set not in visited and (best is None or freq > best[2]):
                    best = (L, ij_set, freq)
                    break
        return best

    def _nodes_to_result(self, nodes: List[Node]) -> Dict[str, Any]:
        roots = [node for node in nodes if node.parent is None]
        result = {
            "selected_nodes": [
                {
                    "L": node.L,
                    "ij_set": sorted(list(node.ij_set)),
                    "frequency": node.frequency,
                    "score": node.get_score(),
                    "parent_L": node.parent.L if node.parent else None,
                }
                for node in nodes
            ],
            "tree_roots": roots,
        }
        logger.info("Selected %d draft nodes across %d tree roots", len(nodes), len(roots))
        for idx, root in enumerate(roots, 1):
            logger.info("  Tree %d: %d nodes", idx, count_tree_nodes(root))
        return result

    def select_budget_limited_drafts(
        self,
        sequence_history: List[Tuple[int, List[Tuple[int, int]]]],
    ) -> Dict[str, Any]:
        """Select the top observed draft formulas per lookahead under the budget."""
        drafts_by_L: Dict[int, Counter] = defaultdict(Counter)
        for L, ij_list in sequence_history:
            if ij_list:
                drafts_by_L[L][frozenset(ij_list)] += 1

        selected_nodes: List[Dict[str, Any]] = []
        total_selected = 0
        for L in sorted(drafts_by_L.keys()):
            if total_selected >= self.budget_B:
                break
            for ij_set_frozen, frequency in drafts_by_L[L].most_common(self.per_step_limit):
                if total_selected >= self.budget_B:
                    break
                selected_nodes.append(
                    {
                        "ij_set": sorted(list(ij_set_frozen)),
                        "frequency": frequency,
                        "score": frequency,
                        "L": L,
                        "parent_L": None,
                    }
                )
                total_selected += 1
                logger.info("Selected draft L=%s freq=%s ij_set=%s", L, frequency, list(ij_set_frozen))
        return {"selected_nodes": selected_nodes, "tree_roots": []}

    def run_calibration(self) -> Dict[str, Any]:
        logger.info("Running calibration for model=%s set=%s", self.model_key, self.calibration_set)
        calibration_data, sequence_history = self.collect_calibration_data()
        possibilities = self.construct_all_possibilities_graph(calibration_data, self.depth_D)
        if self.depth_D * self.per_step_limit > self.budget_B:
            result = self.select_optimal_subgraph(
                possibilities,
                budget_B=self.budget_B,
                sequence_history=sequence_history,
            )
        else:
            result = self.select_budget_limited_drafts(sequence_history)

        result["metadata"] = {
            "model_key": self.model_key,
            "model_family": self.model_family,
            "pretrained_path": self.pretrained_path,
            "calibration_set": self.calibration_set,
            "calibration_sources": self.calibration_sources,
            "disjointness_check": self.disjointness_check,
            "num_samples": self.num_samples,
            "max_seq_len": self.max_seq_len,
            "gen_length": self.gen_length,
            "block_length": self.block_length,
            "depth_D": self.depth_D,
            "budget_B": self.budget_B,
            "per_step_limit": self.per_step_limit,
            "temperature": self.temperature,
            "calibration_tuples": len(calibration_data),
            "sequence_history_entries": len(sequence_history),
        }
        return result
