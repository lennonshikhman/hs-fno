from __future__ import annotations
import warnings

# Legacy result/checkpoint names from the pre-HS-FNO codebase.
MODEL_RENAME = {
    "ablation_hsno_fno_backbone": "hs_fno",
    "hsno": "hsno_unet",
    "ablation_hsno_transformer_backbone": "hs_transformer",
    # Verified from the old runner: no-shift used HistoryToHistoryNO with the
    # default conv/U-Net backbone, and full used hsno_full with the default
    # conv/U-Net backbone, not FNO.
    "ablation_hsno_no_shift": "hsno_unet_no_shift",
    "ablation_hsno_full": "hsno_unet_rollout_semiflow",
    "ablation_hsno_coordinates": "hs_fno_coord_conditioning",
    "ablation_hsno_film": "hs_fno_film_conditioning",
    "ablation_delay_conditioning_off": "hs_fno_no_delay_conditioning",
    "ablation_history_steps_4": "hs_fno_history_steps_4",
    "ablation_history_steps_8": "hs_fno_history_steps_8",
    "ablation_history_steps_12": "hs_fno_history_steps_12",
    "ablation_history_resolution_4": "hs_fno_history_resolution_4",
    "ablation_history_resolution_8": "hs_fno_history_resolution_8",
    "ablation_history_resolution_12": "hs_fno_history_resolution_12",
    "ablation_per_delay_low": "hs_fno_per_delay_low",
    "ablation_per_delay_high": "hs_fno_per_delay_high",
}

LEGACY_MODEL_NAMES = {v: k for k, v in MODEL_RENAME.items()}

CORE_MODELS = [
    "current_state",
    "lag_stack",
    "history2history",
    "temporal_unet",
    "convlstm",
    "temporal_transformer",
    "hs_fno",
]

ABLATION_MODELS = [
    "hs_fno_no_shift",
    "hs_fno_no_delay_conditioning",
    "hs_fno_coord_conditioning",
    "hs_fno_film_conditioning",
    "hs_fno_rollout_semiflow",
    "hs_fno_history_steps_4",
    "hs_fno_history_steps_8",
    "hs_fno_history_steps_12",
    "hs_fno_history_resolution_4",
    "hs_fno_history_resolution_8",
    "hs_fno_history_resolution_12",
    "hs_fno_per_delay_low",
    "hs_fno_per_delay_high",
    "hs_transformer",
    "hsno_unet",
]


def canonical_model_name(name: str, warn: bool = False) -> str:
    out = MODEL_RENAME.get(name, name)
    if warn and out != name:
        warnings.warn(f"Legacy model name '{name}' is deprecated; using '{out}'.", DeprecationWarning, stacklevel=2)
    return out


def rename_model_series(series):
    return series.map(lambda x: MODEL_RENAME.get(str(x), str(x)))
