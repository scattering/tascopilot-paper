from .add_scans import add_scans
from .apply_detailed_balance import apply_detailed_balance
from .calculate_ub import calculate_ub
from .check_resolution import check_resolution
from .check_spurions import check_spurions
from .catalog_bt7_directory import catalog_bt7_directory
from .catalog_tas_directory import catalog_tas_directory
from .catalog_report import catalog_report
from .catalog_tas_batch import catalog_tas_batch
from .experiment_brief import experiment_brief
from .experiment_dashboard import build_experiment_dashboard, survey_inelastic_spurion_evidence
from .experiment_planning_brief import experiment_planning_brief
from .experiment_workspace import experiment_workspace
from .classify_tas_format import classify_tas_format
from .find_experiments import find_experiments
from .find_papers import find_papers
from .group_bt7_scans import group_bt7_scans
from .inspect_bt7_directory import inspect_bt7_directory
from .normalize_to_monitor import normalize_to_monitor
from .nice_bt7_replay import nice_bt7_replay
from .plan_spinwave import plan_spinwave
from .read_bt7_file import read_bt7_file
from .read_tas_metadata import read_tas_metadata
from .real_data_smoke import real_data_smoke
from .retrieve_similar_scans import retrieve_similar_scans
from .scan_tas_directory import scan_tas_directory
from .scan_brief import scan_brief
from .summarize_psd_facility import summarize_psd_facility
from .subtract_background import subtract_background
from .tasai_adaptive_nice_sim import tasai_adaptive_nice_sim
from .translate_ice_file import translate_ice_file
from .trajectory_clustering import cluster_order_parameter_trajectories


def copilot_ask(*args, **kwargs):
    from .copilot_router import copilot_ask as _copilot_ask

    return _copilot_ask(*args, **kwargs)

__all__ = [
    "add_scans",
    "apply_detailed_balance",
    "calculate_ub",
    "check_resolution",
    "check_spurions",
    "catalog_bt7_directory",
    "catalog_tas_directory",
    "catalog_report",
    "catalog_tas_batch",
    "copilot_ask",
    "experiment_brief",
    "build_experiment_dashboard",
    "experiment_planning_brief",
    "experiment_workspace",
    "classify_tas_format",
    "find_experiments",
    "find_papers",
    "group_bt7_scans",
    "inspect_bt7_directory",
    "normalize_to_monitor",
    "nice_bt7_replay",
    "plan_spinwave",
    "read_bt7_file",
    "read_tas_metadata",
    "real_data_smoke",
    "retrieve_similar_scans",
    "scan_tas_directory",
    "scan_brief",
    "summarize_psd_facility",
    "survey_inelastic_spurion_evidence",
    "subtract_background",
    "tasai_adaptive_nice_sim",
    "translate_ice_file",
    "cluster_order_parameter_trajectories",
]
