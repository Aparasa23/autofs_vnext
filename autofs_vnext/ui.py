from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from autofs_vnext.core.registry import REGISTRY, autodiscover
from autofs_vnext.core.runner import run_autofs
from autofs_vnext.core.schemas import AutoFSConfig


def _method_availability(meta) -> Tuple[bool, List[str]]:
    """Check whether a method's optional requirements are importable."""
    import importlib

    missing: List[str] = []
    for req in getattr(meta, "requires", []) or []:
        try:
            importlib.import_module(req)
        except Exception:
            missing.append(req)
    return (len(missing) == 0, missing)


def _available_methods(task: str) -> Tuple[List[str], List[str], List[str]]:
    """Return (cheap_or_medium, expensive, unavailable)."""
    autodiscover()
    avail: List[str] = []
    expensive: List[str] = []
    unavailable: List[str] = []

    for meta in sorted(REGISTRY.list(), key=lambda m: m.name):
        # Filter by task
        if task not in meta.tasks and "classification" in meta.tasks and "regression" in meta.tasks:
            pass
        if task not in meta.tasks and "classification" not in meta.tasks and "regression" not in meta.tasks:
            # Should not happen; defensively skip
            continue
        if task not in meta.tasks:
            continue

        ok, miss = _method_availability(meta)
        if not ok:
            unavailable.append(f"{meta.name} (missing: {', '.join(miss)})")
            continue

        if meta.compute == "expensive":
            expensive.append(meta.name)
        else:
            avail.append(meta.name)

    return avail, expensive, unavailable


def _safe_run_name(s: str) -> str:
    s = (s or "run").strip()
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    out = "".join(keep)
    return out[:64] if out else "run"


def _zip_dir(dir_path: Path) -> str:
    # Returns path to zip file
    base = dir_path.parent / dir_path.name
    zip_path = shutil.make_archive(str(base), "zip", root_dir=str(dir_path))
    return zip_path


def build_app():
    """Build and return the Gradio Blocks app."""
    import gradio as gr

    autodiscover()

    with gr.Blocks(theme=gr.themes.Soft(), title="AutoFS vNext") as demo:
        gr.Markdown(
            """
            # AutoFS vNext – Feature Selection

            Configure a Phase 1/Phase 2 feature selection run and review outputs.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                data_file = gr.File(label="Upload dataset (CSV)", file_types=[".csv"], type="filepath")
                use_example = gr.Checkbox(value=True, label="Use built-in example dataset (ignores upload)")
                target_col = gr.Textbox(value="target", label="Target column")
                task = gr.Radio(choices=["classification", "regression"], value="classification", label="Task")

                run_name = gr.Textbox(value="ui_run", label="Run name")
                max_rows = gr.Number(value=10000, precision=0, label="Row cap (optional)")

                with gr.Accordion("Phase configuration", open=True):
                    phase1_top_k = gr.Number(value=50, precision=0, label="Phase 1: keep top_k (optional)")
                    phase2_top_k = gr.Number(value=50, precision=0, label="Phase 2: keep top_k (optional)")

                    phase1_methods = gr.Dropdown(multiselect=True, label="Phase 1 methods")
                    phase2_methods = gr.Dropdown(multiselect=True, label="Phase 2 methods")

                with gr.Accordion("Aggregation and selection", open=True):
                    agg_strategy = gr.Dropdown(choices=["rrf", "borda"], value="rrf", label="Aggregation")
                    rrf_k = gr.Number(value=60, precision=0, label="RRF k")
                    select_policy = gr.Dropdown(
                        choices=["top_k", "score_threshold", "stability_threshold"],
                        value="top_k",
                        label="Selection policy",
                    )
                    select_top_k = gr.Number(value=30, precision=0, label="Final top_k")
                    score_threshold = gr.Number(value=None, label="Score threshold (if applicable)")
                    stability_threshold = gr.Number(value=0.6, label="Stability threshold (if applicable)")

                with gr.Accordion("Stability and redundancy", open=False):
                    stab_enabled = gr.Checkbox(value=True, label="Enable stability selection")
                    stab_resamples = gr.Slider(5, 50, value=20, step=1, label="Stability resamples")
                    stab_frac = gr.Slider(0.3, 1.0, value=0.8, step=0.05, label="Sample fraction")
                    stab_base = gr.Dropdown(
                        choices=["l1_logistic", "l1_linear", "rf_importance_clf", "rf_importance_reg"],
                        value="l1_logistic",
                        label="Base method",
                    )
                    use_stability = gr.Checkbox(value=True, label="Blend stability into final score")
                    stability_weight = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="Stability weight")

                    red_enabled = gr.Checkbox(value=False, label="Enable correlation clustering")
                    red_threshold = gr.Slider(0.7, 0.99, value=0.9, step=0.01, label="Correlation threshold")
                    redundancy_penalty = gr.Checkbox(value=False, label="Apply redundancy penalty in ranking")
                    redundancy_weight = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Redundancy weight")

                refresh = gr.Button("Refresh method lists")
                run_btn = gr.Button("Run AutoFS", variant="primary")

                unavailable_box = gr.Textbox(label="Unavailable methods (missing optional packages)", lines=6)

            with gr.Column(scale=3):
                status = gr.Markdown("Ready.")
                out_path = gr.Textbox(label="Output directory")
                download_zip = gr.File(label="Download run artifacts")

                selected_df = gr.Dataframe(label="Selected features", interactive=False)
                ranking_df = gr.Dataframe(label="Feature ranking (top 200)", interactive=False)
                metadata = gr.JSON(label="Run metadata")
                log = gr.Textbox(label="Log", lines=10)

        def _load_methods(task_value: str):
            cheap, expensive, unavailable = _available_methods(task_value)
            # Provide sensible defaults
            p1_default = [m for m in cheap if m.startswith("univariate") or m in ("variance", "vif_filter")][:4]
            p2_default = [m for m in ("rfe_logistic", "rf_importance_clf", "stability_selection") if m in cheap + expensive][:3]
            if task_value == "regression":
                p2_default = [m for m in ("bayesian_ridge_importance", "rfe_lasso", "rf_importance_reg") if m in cheap + expensive]

            return (
                gr.Dropdown(choices=cheap, value=p1_default, multiselect=True),
                gr.Dropdown(choices=expensive + cheap, value=p2_default, multiselect=True),
                "\n".join(unavailable) if unavailable else "(all methods available)",
            )

        def _run(
            file_path: str | None,
            use_ex: bool,
            target: str,
            task_value: str,
            run_name_value: str,
            max_rows_value: float | None,
            p1_k: float | None,
            p2_k: float | None,
            p1_methods: List[str],
            p2_methods: List[str],
            agg: str,
            rrf_k_value: float,
            sel_policy: str,
            sel_top_k_value: float,
            score_thr: float | None,
            stab_thr: float | None,
            s_enabled: bool,
            s_resamples: int,
            s_frac: float,
            s_base: str,
            use_stab: bool,
            stab_w: float,
            r_enabled: bool,
            r_thr: float,
            r_penalty: bool,
            r_w: float,
            progress=None,
        ):
            import gradio as gr

            # Determine dataset
            if use_ex:
                dataset_path = Path(__file__).resolve().parents[1] / "examples" / "sample_classification.csv"
            else:
                if not file_path:
                    raise gr.Error("Please upload a CSV or enable the example dataset.")
                dataset_path = Path(file_path)

            # Copy dataset into a temporary working directory to avoid temp cleanup issues
            work_dir = Path(tempfile.mkdtemp(prefix="autofs_ui_"))
            data_copy = work_dir / dataset_path.name
            shutil.copy2(str(dataset_path), str(data_copy))

            # Build config
            rn = _safe_run_name(run_name_value)
            out_dir = work_dir / "autofs_out"

            def _num_or_none(v) -> int | None:
                if v is None:
                    return None
                try:
                    iv = int(v)
                    return iv if iv > 0 else None
                except Exception:
                    return None

            p1_topk = _num_or_none(p1_k)
            p2_topk = _num_or_none(p2_k)
            max_rows_int = _num_or_none(max_rows_value)
            rrf_k_int = max(1, int(rrf_k_value))
            sel_topk_int = max(1, int(sel_top_k_value))

            cfg_dict: Dict[str, Any] = {
                "task": task_value,
                "input": {
                    "path": str(data_copy),
                    "format": "csv",
                    "target": target,
                    "drop_columns": [],
                },
                "preprocess": {
                    "numeric_impute": "median",
                    "categorical_impute": "most_frequent",
                    "scale_numeric": True,
                    "one_hot": True,
                },
                "cv": {
                    "scheme": "stratified_kfold" if task_value == "classification" else "kfold",
                    "n_splits": 5,
                    "shuffle": True,
                    "random_state": 42,
                    "group_column": None,
                },
                "budget": {
                    "max_rows": max_rows_int,
                },
                "phases": [
                    {
                        "name": "Phase_1",
                        "top_k": p1_topk,
                        "methods": [{"name": m, "enabled": True, "params": {}} for m in (p1_methods or [])],
                    },
                    {
                        "name": "Phase_2",
                        "top_k": p2_topk,
                        "methods": [{"name": m, "enabled": True, "params": {}} for m in (p2_methods or [])],
                    },
                ],
                "stability": {
                    "enabled": bool(s_enabled),
                    "n_resamples": int(s_resamples),
                    "sample_frac": float(s_frac),
                    "base_method": s_base,
                    "base_params": {},
                },
                "redundancy": {
                    "enabled": bool(r_enabled),
                    "corr_method": "pearson",
                    "corr_threshold": float(r_thr),
                },
                "aggregation": {
                    "strategy": agg,
                    "rrf_k": rrf_k_int,
                    "use_stability": bool(use_stab),
                    "stability_weight": float(stab_w),
                    "redundancy_penalty": bool(r_penalty),
                    "redundancy_weight": float(r_w),
                },
                "selection": {
                    "policy": sel_policy,
                    "top_k": sel_topk_int,
                    "score_threshold": score_thr,
                    "stability_threshold": stab_thr,
                },
                "output": {
                    "dir": str(out_dir),
                    "run_name": rn,
                },
            }

            cfg = AutoFSConfig.from_dict(cfg_dict)

            # Progress callback
            def _cb(p: float, msg: str):
                if progress is not None:
                    progress(p, desc=msg)

            # Run
            log_lines: List[str] = []
            log_lines.append(f"Working directory: {work_dir}")
            log_lines.append(f"Dataset: {data_copy}")
            log_lines.append(f"Task: {task_value}, target: {target}")
            log_lines.append(f"Phase_1 methods: {p1_methods}")
            log_lines.append(f"Phase_2 methods: {p2_methods}")

            out_path = run_autofs(cfg, progress_cb=_cb)

            # Load artifacts
            selected_path = out_path / "selected_features.csv"
            ranking_path_csv = out_path / "feature_ranking.csv"
            ranking_path_parquet = out_path / "feature_ranking.parquet"
            meta_path = out_path / "run_metadata.json"

            selected = pd.read_csv(selected_path)
            if ranking_path_parquet.exists():
                try:
                    ranking = pd.read_parquet(ranking_path_parquet)
                except Exception:
                    ranking = pd.read_csv(ranking_path_csv)
            else:
                ranking = pd.read_csv(ranking_path_csv)

            top_ranking = ranking.head(200)
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

            zip_file = _zip_dir(out_path)
            log_lines.append(f"Output: {out_path}")
            log_lines.append(f"Artifacts zip: {zip_file}")

            return (
                f"### Completed: `{rn}`\nArtifacts are ready below.",
                str(out_path),
                zip_file,
                selected,
                top_ranking,
                meta,
                "\n".join(log_lines),
            )

        refresh.click(
            fn=_load_methods,
            inputs=[task],
            outputs=[phase1_methods, phase2_methods, unavailable_box],
        )
        task.change(
            fn=_load_methods,
            inputs=[task],
            outputs=[phase1_methods, phase2_methods, unavailable_box],
        )

        run_btn.click(
            fn=_run,
            inputs=[
                data_file,
                use_example,
                target_col,
                task,
                run_name,
                max_rows,
                phase1_top_k,
                phase2_top_k,
                phase1_methods,
                phase2_methods,
                agg_strategy,
                rrf_k,
                select_policy,
                select_top_k,
                score_threshold,
                stability_threshold,
                stab_enabled,
                stab_resamples,
                stab_frac,
                stab_base,
                use_stability,
                stability_weight,
                red_enabled,
                red_threshold,
                redundancy_penalty,
                redundancy_weight,
            ],
            outputs=[status, out_path, download_zip, selected_df, ranking_df, metadata, log],
            show_progress=True,
        )

        # Initialize method lists
        demo.load(fn=_load_methods, inputs=[task], outputs=[phase1_methods, phase2_methods, unavailable_box])

    return demo


def launch(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    app = build_app()
    app.queue(concurrency_count=1)
    app.launch(server_name=host, server_port=port, share=share)
