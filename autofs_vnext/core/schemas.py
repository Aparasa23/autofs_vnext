from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

TaskType = Literal["classification", "regression"]

@dataclass
class InputSpec:
    path: str
    format: Literal["csv", "parquet"] = "csv"
    target: str = ""
    drop_columns: List[str] = field(default_factory=list)

@dataclass
class PreprocessSpec:
    categorical_strategy: Literal["onehot"] = "onehot"
    numeric_impute: Literal["median", "mean"] = "median"
    categorical_impute: Literal["most_frequent"] = "most_frequent"
    scale_numeric: bool = True
    max_ohe_levels: int = 500  # safety for extreme cardinality
    sparse_ohe: bool = True

@dataclass
class CVSpec:
    scheme: Literal["kfold", "stratified_kfold", "group_kfold"] = "kfold"
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    group_column: Optional[str] = None

@dataclass
class BudgetSpec:
    max_rows: Optional[int] = None
    max_features_after_screen: Optional[int] = None
    per_method_timeout_sec: Optional[int] = None  # best-effort (soft) timeout
    n_jobs: int = -1

@dataclass
class PhaseSpec:
    name: str
    methods: List[Dict[str, Any]]  # each: {"name": "...", "params": {...}, "enabled": true}
    top_k: Optional[int] = None
    score_threshold: Optional[float] = None

@dataclass
class AggregationSpec:
    strategy: Literal["rrf", "borda"] = "rrf"
    rrf_k: int = 60
    use_stability: bool = True
    stability_weight: float = 0.35
    redundancy_penalty: bool = False
    redundancy_weight: float = 0.15

@dataclass
class SelectionSpec:
    policy: Literal["top_k", "score_threshold", "stability_threshold"] = "top_k"
    top_k: int = 100
    score_threshold: Optional[float] = None
    stability_threshold: Optional[float] = None

@dataclass
class RedundancySpec:
    enabled: bool = False
    corr_method: Literal["pearson", "spearman"] = "pearson"
    corr_threshold: float = 0.95
    representative: Literal["best_ranked", "top_n", "annotate_only"] = "best_ranked"
    top_n: int = 1

@dataclass
class StabilitySpec:
    enabled: bool = True
    n_resamples: int = 20
    sample_frac: float = 0.75
    base_method: str = "l1_logistic"  # or "l1_linear"
    base_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OutputSpec:
    dir: str = "autofs_out"
    run_name: str = "run"

@dataclass
class AutoFSConfig:
    task: TaskType
    input: InputSpec
    preprocess: PreprocessSpec = field(default_factory=PreprocessSpec)
    cv: CVSpec = field(default_factory=CVSpec)
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    phases: List[PhaseSpec] = field(default_factory=list)
    aggregation: AggregationSpec = field(default_factory=AggregationSpec)
    selection: SelectionSpec = field(default_factory=SelectionSpec)
    stability: StabilitySpec = field(default_factory=StabilitySpec)
    redundancy: RedundancySpec = field(default_factory=RedundancySpec)
    output: OutputSpec = field(default_factory=OutputSpec)

def _coerce_dataclass(cls, d: Dict[str, Any]):
    # Minimalistic loader without external deps
    fields = {f.name: f for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {}
    for k, v in d.items():
        if k not in fields:
            continue
        ft = fields[k].type
        kwargs[k] = v
    return cls(**kwargs)  # type: ignore[misc]

def load_config(path: str) -> AutoFSConfig:
    import json
    from pathlib import Path
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = AutoFSConfig(
        task=raw["task"],
        input=_coerce_dataclass(InputSpec, raw["input"]),
        preprocess=_coerce_dataclass(PreprocessSpec, raw.get("preprocess", {})),
        cv=_coerce_dataclass(CVSpec, raw.get("cv", {})),
        budget=_coerce_dataclass(BudgetSpec, raw.get("budget", {})),
        aggregation=_coerce_dataclass(AggregationSpec, raw.get("aggregation", {})),
        selection=_coerce_dataclass(SelectionSpec, raw.get("selection", {})),
        stability=_coerce_dataclass(StabilitySpec, raw.get("stability", {})),
        redundancy=_coerce_dataclass(RedundancySpec, raw.get("redundancy", {})),
        output=_coerce_dataclass(OutputSpec, raw.get("output", {})),
        phases=[],
    )
    phases = []
    for ph in raw.get("phases", []):
        phases.append(_coerce_dataclass(PhaseSpec, ph))
    cfg.phases = phases
    return cfg
