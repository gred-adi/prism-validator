def extract_numeric(value):
        try:
            return float(str(value).split(', ')[1])
        except (IndexError, ValueError):
            return float('nan')

import os
import ast
import shutil
import warnings
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s -%(message)s",
    handlers=[
        logging.FileHandler("report_generator.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

import click

from utils.qa_reporting import (
    generate_qa_report,
    generate_sprint_summary_report
)
from utils.qa_ks_comparison import compare_data_distributions
from utils.qa_plotting import (
    generate_report_plots,
    generate_summary_fprp_plots,
)
from utils.qa_reporting import generate_summary_fprp

warnings.filterwarnings("ignore")

def build_partial_config_table(base_folder: str, site: str, asset: str,
                               sub_ts_length: int, n_ts_above_thresh: int,
                               time_interval: int, warning: float, alert: float) -> pd.DataFrame:
    rows = []
    base_path = Path(base_folder)
    sprint_dirs = [d for d in base_path.iterdir() if d.is_dir() and 'sprint' in d.name.lower()]

    for sprint_dir in sprint_dirs:
        sprint_name = sprint_dir.name
        for model_dir in sprint_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            dataset_path = model_dir / "dataset"

            if not dataset_path.is_dir():
                logging.warning(f"Skipping {model_name} in {sprint_name}: no dataset folder found")
                continue

            raw_files = [f.name for f in dataset_path.iterdir()
                         if f.is_file() and 'RAW' in f.name.upper()
                         and f.name.lower().endswith('.csv')]
            raw_fname = raw_files[0] if raw_files else ""

            holdout_files = [f.name for f in dataset_path.iterdir()
                             if f.is_file() and 'HOLDOUT' in f.name.upper()
                             and f.name.lower().endswith('.csv')]
            holdout_fname = holdout_files[0] if holdout_files else ""

            if not raw_fname or not holdout_fname:
                logging.warning(f"Skipping model {model_name} in sprint {sprint_name}: missing raw or holdout data file")
                continue

            rows.append({
                "site": site,
                "asset": asset,
                "sprint_name": sprint_name,
                "model_name": model_name,
                "raw_data_fname": raw_fname,
                "holdout_data_fname": holdout_fname,
                "sub_ts_length_in_minutes": sub_ts_length,
                "n_ts_above_threshold": n_ts_above_thresh,
                "time_interval": time_interval,
                "warning_threshold": warning,
                "alert_threshold": alert
            })

    df = pd.DataFrame(rows)
    if df.empty:
        logging.warning(f"No valid models found in {base_folder}. Returning an empty DataFrame.")
    return df


def merge_constraints(partial_df: pd.DataFrame, constraints_csv: str) -> pd.DataFrame:
    """
    Read a one-constraint-per-row lookup CSV, aggregate into list columns,
    and merge into the partial config table.
    """

    constraints = pd.read_csv(constraints_csv)

    agg = (
        constraints
        .groupby("model_name", as_index=False)
        .agg({
            "constraint_cols":    list,
            "operators":          list,
            "constraint_limits":  list
        })
    )

    full_df = partial_df.merge(
        agg,
        on="model_name",
        how="left"
    )

    for col in ["constraint_cols", "operators", "constraint_limits"]:
        full_df[col] = full_df[col].apply(lambda x: x if isinstance(x, list) else [])

    missing = full_df[full_df["constraint_cols"].apply(lambda x: not x)]["model_name"].tolist()
    if missing:
        logging.warning(f"These models have no constraints: {', '.join(missing)}")

    return full_df


class ReportGenerator:
    """Drive the QA pipeline entirely from one flat CSV table."""

    def __init__(self, table_path: str, local_path: str, regenerate: bool = False):
        self.df = pd.read_csv(
            table_path,
            converters={
                "constraint_cols": ast.literal_eval,
                "operators": ast.literal_eval,
                "constraint_limits": ast.literal_eval
            }
        )
        # Added by Hans, removes null columns
        required_cols = ["sprint_name", "model_name", "raw_data_fname", "holdout_data_fname"]
        self.df = self.df.dropna(subset=required_cols)
        self.local_path = local_path
        self.regenerate = regenerate

        if not (self.df[["time_interval", "sub_ts_length_in_minutes", "warning_threshold", "alert_threshold"]].nunique() == 1).all():
            raise ValueError("Inconsistent global settings across rows in table")

        first = self.df.iloc[0]
        ti = first["time_interval"]
        self.time_interval = ti
        self.sub_ts_length = first["sub_ts_length_in_minutes"] / ti
        self.n_ts_above_thresh = first["n_ts_above_threshold"] / ti
        self.warning_threshold = first["warning_threshold"]
        self.alert_threshold = first["alert_threshold"]
        self.number_of_sprints = int(first.get("number_of_sprints", 0))

        self.summary_base = str(Path(self.local_path) / "Summary_Reports")
        Path(self.summary_base).mkdir(parents=True, exist_ok=True)
        self.missing_models: List[str] = []

    def process_sprints(self) -> None:
        """Process each model in the configuration table and generate sprint summaries."""
        for _, row in self.df.iterrows():
            if any(pd.isna(row.get(col)) for col in ["sprint_name", "model_name", "raw_data_fname", "holdout_data_fname"]):
                logging.warning(f"Skipping incomplete row: {row}")
                continue
            model = row.get("model_name", "<unknown>")
            try:
                self._process_model(row)
            except FileNotFoundError as e:
                logging.error(f"{model} directory cannot be found: {e}. Moving to next model.")
                self.missing_models.append(model)
                continue

        for sprint, _ in self.df.groupby("sprint_name"):
            sprint_path = Path(self.local_path) / sprint
            if not sprint_path.is_dir():
                logging.warning(f"{sprint} directory not found, skipping summaries.")
                continue
            self._generate_sprint_summaries(str(sprint_path))

        self._gather_all_reports()

        if self.missing_models:
            logging.info("The following models were skipped because their folders were missing:")
            for m in self.missing_models:
                logging.info(f" - {m}")

    def _process_model(self, row: pd.Series) -> None:
        sprint = row["sprint_name"]
        model = row["model_name"]
        raw = row["raw_data_fname"]
        hold = row["holdout_data_fname"]
        cols = row["constraint_cols"]
        limits = row["constraint_limits"]
        ops = row["operators"]

        base_dir = Path(self.local_path) / sprint / model
        if not base_dir.is_dir():
            raise FileNotFoundError(f"Model folder not found: {base_dir}")

        logging.info(f"> Generating report for model: {model}")
        paths = self._get_file_paths(sprint, model, raw, hold)
        for key in ("fpr_path", "ks_path", "report_path"):
            Path(paths[key]).mkdir(parents=True, exist_ok=True)

        if self.regenerate or not any(f.endswith(".jpg") for f in os.listdir(paths["fpr_path"])):
            self._generate_fpr_plots(model, paths, cols, limits, ops)
        if self.regenerate or not any(f.endswith(".jpg") for f in os.listdir(paths["ks_path"])):
            self._generate_ks_plots(paths)
        if self.regenerate or not any(f.endswith(".pdf") for f in os.listdir(paths["report_path"])):
            self._generate_model_report(model, paths)

    def _get_file_paths(
        self,
        sprint: str,
        model: str,
        raw_fname: str,
        hold_fname: str
    ) -> Dict[str, str]:
        """
        Generate file paths for a given sprint and model, validating the existence of critical files.
        """
        base_dir = Path(self.local_path) / sprint / model
        dataset_path = base_dir / "dataset"
        data_splitting_path = base_dir / "data_splitting"
        perf_dir = base_dir / "performance_assessment_report"
        fpr_dir = perf_dir / "FPR"
        ks_dir = perf_dir / "KS"

        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Data folders missing under {base_dir}")

        raw_data_path = dataset_path / raw_fname
        holdout_path = dataset_path / hold_fname
        if not raw_data_path.is_file():
            raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")
        if not holdout_path.is_file():
            raise FileNotFoundError(f"Holdout data file not found: {holdout_path}")

        def find_file_with_keywords(folder, *keywords):
            if not folder.is_dir():
                return None
            files = [f.name for f in folder.iterdir() if f.is_file()]
            for k in keywords:
                # Prioritize more specific match
                match = next((f for f in files if k in f.upper()), None)
                if match:
                    return str(folder / match)
            return None

        # files = [f.name for f in dataset_path.iterdir() if f.is_file()]
        # val_omr_wo = next((f for f in files if "OMR-CLEANED-VALIDATION" in f.upper()), None)
        # val_omr_w = next((f for f in files if "OMR-RAW-VALIDATION" in f.upper()), None)
        # hold_omr = next((f for f in files if "OMR-HOLDOUT" in f.upper()), None)
        val_omr_wo = find_file_with_keywords(
            data_splitting_path, "OMR-VALIDATION-WITHOUT-OUTLIER"
        ) or find_file_with_keywords(
            dataset_path, "OMR-VALIDATION-WITHOUT-OUTLIER"
        )

        val_omr_w = find_file_with_keywords(
            data_splitting_path, "OMR-VALIDATION-WITH-OUTLIER"
        ) or find_file_with_keywords(
            dataset_path, "OMR-VALIDATION-WITH-OUTLIER"
        )

        hold_omr = find_file_with_keywords(
            dataset_path, "OMR-HOLDOUT"
        )

        return {
            "dataset_path": str(dataset_path),
            "raw_data_path": str(raw_data_path),
            "holdout_path": str(holdout_path),
            "val_without_outlier_omr": val_omr_wo,
            "val_with_outlier_omr": val_omr_w,
            "holdout_omr": hold_omr,
            "fpr_path": str(fpr_dir),
            "ks_path": str(ks_dir),
            "report_path": str(perf_dir / "report_document"),
            "data_stats_fpath": str(ks_dir / "data_stats.yaml"),
            "datasets_range_fpath": str(fpr_dir / "datasets_range.yaml"),
            "ks_df_fpath": str(ks_dir / "ks_results.csv"),
            "fpr_stats_cleaned_omr_fpath": str(fpr_dir / "fpr_stats_cleaned_val_omr_df.yaml"),
            "fpr_stats_holdout_omr_fpath": str(fpr_dir / "fpr_stats_holdout_omr_df.yaml"),
            "fpr_stats_raw_omr_fpath": str(fpr_dir / "fpr_stats_raw_val_omr_df.yaml"),
            "fprp_stats_holdout_omr_fpath": str(fpr_dir / "fprp_stats_holdout_omr_df.yaml")
        }

    def _generate_fpr_plots(self, model_name: str, paths: Dict[str, str], cols: List[str], limits: List[Any], ops: List[str]) -> None:
        generate_report_plots(
            data_fpath                    = paths["dataset_path"],
            fpr_fpath                     = paths["fpr_path"],
            model_name                    = model_name,
            constraint_cols               = cols,
            condition_limits              = limits,
            operators                     = ops,
            raw_data_fpath                = paths["raw_data_path"],
            holdout_fpath                 = paths["holdout_path"],
            holdout_omr_fname             = paths["holdout_omr"],
            val_without_outlier_omr_fname = paths["val_without_outlier_omr"],
            val_with_outlier_omr_fname    = paths["val_with_outlier_omr"],
            sub_ts_length_in_minutes      = self.sub_ts_length,
            n_ts_above_threshold          = self.n_ts_above_thresh,
            time_interval                 = self.time_interval,
            warning_threshold             = self.warning_threshold,
            alert_threshold               = self.alert_threshold,
            nsteps                        = 20
        )

    def _generate_ks_plots(self, paths: Dict[str, str]) -> None:
        compare_data_distributions(
            validation_fname = paths["raw_data_path"],
            ks_file_path     = paths["ks_path"],
            holdout_fpath    = paths["holdout_path"],
            description_row = 0
        )

    def _generate_model_report(self, model_name: str, paths: Dict[str, str]) -> None:
        generate_qa_report(
            model_name                    = model_name,
            fpr_file_path                 = paths["fpr_path"],
            ks_file_path                  = paths["ks_path"],
            report_file_path              = paths["report_path"],
            fpr_stats_cleaned_omr_fpath   = paths["fpr_stats_cleaned_omr_fpath"],
            fpr_stats_holdout_omr_fpath   = paths["fpr_stats_holdout_omr_fpath"],
            fpr_stats_raw_omr_fpath       = paths["fpr_stats_raw_omr_fpath"],
            fprp_stats_holdout_omr_fpath  = paths["fprp_stats_holdout_omr_fpath"],
            data_stats_fpath              = paths["data_stats_fpath"],
            ks_df_fpath                   = paths["ks_df_fpath"],
            datasets_range_fpath          = paths["datasets_range_fpath"]
        )

    def _generate_sprint_summaries(self, sprint_path: str) -> None:
        fprp_df = generate_summary_fprp(sprint_path)
        generate_summary_fprp_plots(
            df         = fprp_df,
            plot_fname = os.path.join(sprint_path, "fprp_plots.jpg"),
            fpr_limit  = 100
        )
        generate_sprint_summary_report(
            sprint_name = os.path.basename(sprint_path),
            sprint_path = sprint_path
        )

    def _gather_all_reports(self) -> None:
        for sprint, group in self.df.groupby("sprint_name"):
            logging.info(f"> Collecting reports from {sprint}")
            dst_folder = Path(self.summary_base) / sprint
            dst_folder.mkdir(parents=True, exist_ok=True)

            summary_pdf = Path(self.local_path) / sprint / f"{sprint} Summary Results.pdf"
            if summary_pdf.exists():
                shutil.copy(summary_pdf, dst_folder)

            for model in group["model_name"]:
                report_dir = Path(self.local_path) / sprint / model / "performance_assessment_report" / "report_document"
                if not report_dir.is_dir():
                    logging.warning(f"{model} report directory not found, skipping.")
                    continue
                for fn in report_dir.iterdir():
                    if fn.suffix.lower() == ".pdf":
                        shutil.copy(fn, dst_folder)


def generate_report_from_table(
    table_path: str,
    local_path: str,
    regenerate: bool = False
) -> None:
    rg = ReportGenerator(table_path, local_path, regenerate)
    rg.process_sprints()
