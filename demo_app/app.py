from __future__ import annotations

import io
import json
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from demo_utils import (
        ensure_dir,
        read_json,
        recompute_row_with_controls,
        repo_root,
        resolve_mode,
        scripted_steps,
        summarize_export_caption,
        timestamp_slug,
    )
    from model_adapters.poster_mode import PosterModeAdapter
except ModuleNotFoundError:
    from demo_app.demo_utils import (
        ensure_dir,
        read_json,
        recompute_row_with_controls,
        repo_root,
        resolve_mode,
        scripted_steps,
        summarize_export_caption,
        timestamp_slug,
    )
    from demo_app.model_adapters.poster_mode import PosterModeAdapter


ROOT = repo_root()
ASSETS_DIR = ROOT / "demo_assets"
OUTPUTS_DIR = ROOT / "demo_outputs"
POSTER_DIR = ROOT / "final_isef_package"
EXPORT_ROOT = POSTER_DIR / "demo_exports"
MODELS_DIR = ROOT / "models"


def _default_mode_from_env() -> str:
    value = os.environ.get("UAV_DEMO_MODE", "poster").strip().lower()
    if value in {"poster", "full"}:
        return value
    return "poster"


@st.cache_data(show_spinner=False)
def load_reference_tables() -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    tables["metrics_unimodal"] = pd.read_csv(
        POSTER_DIR / "tables" / "metrics_unimodal.csv"
    )
    tables["metrics_fusion"] = pd.read_csv(POSTER_DIR / "tables" / "metrics_fusion.csv")
    tables["ablation"] = pd.read_csv(POSTER_DIR / "tables" / "ablation_table.csv")
    tables["stress_audio"] = pd.read_csv(
        POSTER_DIR / "tables" / "stress_audio_snr.csv"
    )
    tables["stress_vision"] = pd.read_csv(
        POSTER_DIR / "tables" / "stress_vision_corruptions.csv"
    )
    tables["stress_radar"] = pd.read_csv(
        POSTER_DIR / "tables" / "stress_radar_clutter.csv"
    )
    tables["missing_modality"] = pd.read_csv(
        POSTER_DIR / "tables" / "missing_modality.csv"
    )
    tables["preds_fusion"] = pd.read_csv(POSTER_DIR / "tables" / "preds_fusion.csv")
    return tables


@st.cache_data(show_spinner=False)
def load_latency_summary() -> Dict[str, object]:
    return read_json(POSTER_DIR / "tables" / "latency_summary.json")


@st.cache_data(show_spinner=False)
def load_demo_outputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    poster = PosterModeAdapter(OUTPUTS_DIR)
    return poster.predictions, poster.gate_weights, poster.bboxes


@st.cache_data(show_spinner=False)
def load_sample_metadata() -> Dict[str, object]:
    return read_json(ASSETS_DIR / "sample_metadata.json")


@st.cache_data(show_spinner=False)
def load_radar_matrix() -> np.ndarray:
    radar_df = pd.read_csv(ASSETS_DIR / "sample_radar.csv")
    value_cols = [c for c in radar_df.columns if c.startswith("bin_")]
    return radar_df[value_cols].to_numpy(dtype=np.float32)


@st.cache_data(show_spinner=False)
def load_audio_waveform_from_path(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as handle:
        sample_rate = handle.getframerate()
        n_channels = handle.getnchannels()
        n_frames = handle.getnframes()
        raw = handle.readframes(n_frames)
    pcm = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    waveform = pcm.astype(np.float32) / 32768.0
    return waveform, sample_rate


def _apply_judge_css(enabled: bool) -> None:
    if not enabled:
        return
    st.markdown(
        """
<style>
  [data-testid="stSidebar"] {display: none;}
  .block-container {max-width: 1800px; padding-top: 1rem; padding-bottom: 1rem;}
  .judge-title {font-size: 1.45rem; font-weight: 700; line-height: 1.3;}
  .judge-panel {border: 2px solid #1f2937; border-radius: 10px; padding: 0.7rem;}
  .judge-badge {background: #0f766e; color: white; border-radius: 8px; padding: 0.3rem 0.6rem; font-weight: 700;}
</style>
        """,
        unsafe_allow_html=True,
    )


def _load_uploaded_file(uploaded, suffix: str) -> Path:
    ensure_dir(ROOT / ".tmp")
    output_path = ROOT / ".tmp" / f"upload_{timestamp_slug()}{suffix}"
    output_path.write_bytes(uploaded.getbuffer())
    return output_path


def _render_frame_with_bbox(
    video_path: Path, bbox_df: pd.DataFrame, frame_idx: int, scenario: str
) -> Tuple[np.ndarray, int]:
    try:
        import cv2  # type: ignore
    except Exception:
        fallback_img = POSTER_DIR / "verification" / "IR_DRONE_059_f000112_DRONE.png"
        image = plt.imread(str(fallback_img))
        if image.dtype == np.uint8:
            rgb = image[:, :, :3]
        else:
            rgb = np.clip(image[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
        return rgb, 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fallback_img = POSTER_DIR / "verification" / "IR_DRONE_059_f000112_DRONE.png"
        image = plt.imread(str(fallback_img))
        if image.dtype == np.uint8:
            rgb = image[:, :, :3]
        else:
            rgb = np.clip(image[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
        return rgb, 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1
    idx = max(0, min(frame_idx, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rows = bbox_df[(bbox_df["scenario"] == scenario) & (bbox_df["frame_idx"] == idx)]
    if rows.empty:
        rows = bbox_df[(bbox_df["scenario"] == "clean") & (bbox_df["frame_idx"] == idx)]
    for _, row in rows.iterrows():
        x1, y1 = int(row["x1"]), int(row["y1"])
        x2, y2 = int(row["x2"]), int(row["y2"])
        label = str(row["label"])
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 80), 2)
        cv2.putText(
            frame_rgb,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 240, 90),
            2,
            cv2.LINE_AA,
        )
    return frame_rgb, total_frames


def _build_runtime_predictions(
    base_df: pd.DataFrame,
    audio_snr_db: float,
    vision_severity: float,
    radar_clutter_level: float,
    miss_vis: int,
    miss_aud: int,
    miss_rad: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for row in base_df.to_dict(orient="records"):
        updated = recompute_row_with_controls(
            row=row,
            audio_snr_db=audio_snr_db,
            vision_severity=vision_severity,
            radar_clutter_level=radar_clutter_level,
            miss_vis=miss_vis,
            miss_aud=miss_aud,
            miss_rad=miss_rad,
        )
        payload: Dict[str, float] = {
            "frame_idx": float(row["frame_idx"]),
            "t_sec": float(row["t_sec"]),
            "y_true": float(row["y_true"]),
            "p_vis": updated["p_vis"],
            "p_aud": updated["p_aud"],
            "p_rad": updated["p_rad"],
            "p_avg": updated["p_avg"],
            "p_static": updated["p_static"],
            "p_gate": updated["p_gate"],
            "p_gate_noq": updated["p_gate_noq"],
            "w_vis": updated["w_vis"],
            "w_aud": updated["w_aud"],
            "w_rad": updated["w_rad"],
            "audio_snr_db": updated["audio_snr_db"],
            "vision_severity": updated["vision_severity"],
            "radar_clutter_level": updated["radar_clutter_level"],
            "miss_vis": updated["miss_vis"],
            "miss_aud": updated["miss_aud"],
            "miss_rad": updated["miss_rad"],
        }
        rows.append(payload)
    return pd.DataFrame(rows)


def _plot_gating(runtime_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.plot(runtime_df["t_sec"], runtime_df["w_vis"], label="w_vis", linewidth=2.0)
    ax.plot(runtime_df["t_sec"], runtime_df["w_aud"], label="w_aud", linewidth=2.0)
    ax.plot(runtime_df["t_sec"], runtime_df["w_rad"], label="w_rad", linewidth=2.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gate Weight")
    ax.set_title("Condition-Adaptive Gate Weights")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def _plot_confidence(row: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    labels = ["Vision", "Audio", "Radar", "F-AVG", "F-STATIC", "F-GATE"]
    values = [
        float(row["p_vis"]),
        float(row["p_aud"]),
        float(row["p_rad"]),
        float(row["p_avg"]),
        float(row["p_static"]),
        float(row["p_gate"]),
    ]
    colors = ["#1d4ed8", "#16a34a", "#dc2626", "#6b7280", "#111827", "#0f766e"]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Confidence")
    ax.set_title("Per-Modality and Fusion Confidence")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_audio(waveform: np.ndarray, sample_rate: int) -> plt.Figure:
    n = waveform.shape[0]
    t = np.arange(n) / sample_rate
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 3.8), sharex=False)
    axes[0].plot(t, waveform, linewidth=0.7, color="#1d4ed8")
    axes[0].set_title("Audio Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amp")
    axes[0].grid(alpha=0.2)
    axes[1].specgram(waveform, Fs=sample_rate, NFFT=512, noverlap=320, cmap="magma")
    axes[1].set_title("Audio Spectrogram")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Hz")
    fig.tight_layout()
    return fig


def _plot_radar(radar_matrix: np.ndarray, frame_idx: int) -> plt.Figure:
    if radar_matrix.ndim != 2:
        radar_matrix = np.zeros((1, 64), dtype=np.float32)
    idx = int(max(0, min(frame_idx, radar_matrix.shape[0] - 1)))
    frame = radar_matrix[idx].reshape(8, 8)
    fig, ax = plt.subplots(figsize=(4.0, 3.3))
    im = ax.imshow(frame, cmap="viridis", aspect="auto")
    ax.set_title("Radar Feature Heatmap")
    ax.set_xlabel("Doppler Bin")
    ax.set_ylabel("Range Bin")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    fig.tight_layout()
    return fig


def _plot_stress_curves(
    stress_audio: pd.DataFrame, stress_vision: pd.DataFrame, stress_radar: pd.DataFrame
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.0))
    for name, group in stress_audio.groupby("run_id"):
        group = group.sort_values("snr_db")
        axes[0].plot(group["snr_db"], group["pr_auc"], label=name)
    axes[0].set_title("Audio SNR Stress")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("PR-AUC")
    axes[0].grid(alpha=0.25)

    vis = (
        stress_vision.groupby(["run_id", "severity"], as_index=False)["pr_auc"]
        .mean()
        .sort_values("severity")
    )
    for name, group in vis.groupby("run_id"):
        axes[1].plot(group["severity"], group["pr_auc"], label=name)
    axes[1].set_title("Vision Corruption Stress")
    axes[1].set_xlabel("Severity")
    axes[1].set_ylabel("PR-AUC")
    axes[1].grid(alpha=0.25)

    for name, group in stress_radar.groupby("run_id"):
        group = group.sort_values("clutter_level")
        axes[2].plot(group["clutter_level"], group["pr_auc"], label=name)
    axes[2].set_title("Radar Clutter Stress")
    axes[2].set_xlabel("Clutter Level")
    axes[2].set_ylabel("PR-AUC")
    axes[2].grid(alpha=0.25)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    return fig


def _build_export_bundle(
    mode: str,
    scenario: str,
    runtime_df: pd.DataFrame,
    frame_idx: int,
    overlay_frame: np.ndarray,
    controls: Dict[str, object],
) -> Path:
    ensure_dir(EXPORT_ROOT)
    out_dir = EXPORT_ROOT / timestamp_slug()
    ensure_dir(out_dir)

    from PIL import Image

    frame_path = out_dir / "overlay_frame.png"
    Image.fromarray(overlay_frame).save(frame_path)

    gating_fig = _plot_gating(runtime_df)
    gating_path = out_dir / "gating_weights_chart.png"
    gating_fig.savefig(gating_path, dpi=180)
    plt.close(gating_fig)

    row = runtime_df.iloc[int(max(0, min(frame_idx, len(runtime_df) - 1)))]
    conf_fig = _plot_confidence(row)
    conf_path = out_dir / "confidence_panel.png"
    conf_fig.savefig(conf_path, dpi=180)
    plt.close(conf_fig)

    runtime_df[["frame_idx", "t_sec", "w_vis", "w_aud", "w_rad"]].to_csv(
        out_dir / "gating_weights.csv", index=False
    )
    runtime_df[
        [
            "frame_idx",
            "t_sec",
            "p_vis",
            "p_aud",
            "p_rad",
            "p_avg",
            "p_static",
            "p_gate",
            "p_gate_noq",
        ]
    ].to_csv(out_dir / "predictions_per_frame.csv", index=False)

    payload = {
        "mode": mode,
        "scenario": scenario,
        "frame_idx": frame_idx,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "controls": controls,
    }
    with (out_dir / "run_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    caption = summarize_export_caption(payload)
    (out_dir / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    return out_dir


def main() -> None:
    st.set_page_config(
        page_title="ISEF UAV Multimodal Demo",
        page_icon="UAV",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Condition-Adaptive Multimodal UAV Detection Demo")
    st.caption(
        "Vision + Audio + Radar fusion with adaptive gating, missing-modality handling, and poster-ready exports."
    )

    if "judge_mode" not in st.session_state:
        st.session_state.judge_mode = False
    if "script_step_idx" not in st.session_state:
        st.session_state.script_step_idx = 0
    if "last_export_dir" not in st.session_state:
        st.session_state.last_export_dir = ""

    controls_col, badge_col = st.columns([1.2, 3.8])
    toggle_label = "Exit Judge Mode" if st.session_state.judge_mode else "Judge Mode"
    if controls_col.button(toggle_label, use_container_width=True):
        st.session_state.judge_mode = not st.session_state.judge_mode
    _apply_judge_css(st.session_state.judge_mode)

    if st.session_state.judge_mode:
        badge_col.markdown(
            "<span class='judge-badge'>Judge Mode Enabled</span>",
            unsafe_allow_html=True,
        )
    else:
        badge_col.info("Judge Mode Off")

    default_mode = _default_mode_from_env()
    mode_choice = st.sidebar.radio(
        "Runtime Mode",
        options=["poster", "full"],
        index=0 if default_mode == "poster" else 1,
        help="Poster Mode is offline replay from committed outputs; Full Mode requires local weights under models/.",
    )
    mode_info = resolve_mode(mode_choice, ROOT, MODELS_DIR)
    if mode_info.mode == "error":
        st.error(mode_info.details)
        st.stop()
    st.success(mode_info.badge_text)
    st.caption(mode_info.details)

    tables = load_reference_tables()
    preds_demo, _, bboxes = load_demo_outputs()
    metadata = load_sample_metadata()

    scenario_options = sorted(preds_demo["scenario"].unique().tolist())
    live_scenario = st.sidebar.selectbox("Scenario", options=scenario_options, index=0)
    audio_snr_db = st.sidebar.slider("Audio SNR (dB)", -10.0, 30.0, 30.0, 1.0)
    vision_severity = st.sidebar.slider("Vision Degradation", 0.0, 3.0, 0.0, 0.25)
    radar_clutter = st.sidebar.slider("Radar Clutter", 0.0, 1.0, 0.1, 0.05)
    miss_vis = int(st.sidebar.checkbox("Drop Vision", value=False))
    miss_aud = int(st.sidebar.checkbox("Drop Audio", value=False))
    miss_rad = int(st.sidebar.checkbox("Drop Radar", value=False))

    input_source = st.sidebar.radio("Input Source", ["Built-in sample", "Upload custom"])
    video_path = ASSETS_DIR / "sample_video.mp4"
    audio_path = ASSETS_DIR / "sample_audio.wav"
    radar_path = ASSETS_DIR / "sample_radar.csv"

    if input_source == "Upload custom":
        upload_video = st.sidebar.file_uploader("Upload MP4", type=["mp4"])
        upload_audio = st.sidebar.file_uploader("Upload WAV", type=["wav"])
        upload_radar = st.sidebar.file_uploader("Upload radar CSV/NPY", type=["csv", "npy"])
        if upload_video is not None:
            video_path = _load_uploaded_file(upload_video, ".mp4")
        if upload_audio is not None:
            audio_path = _load_uploaded_file(upload_audio, ".wav")
        if upload_radar is not None:
            radar_path = _load_uploaded_file(
                upload_radar, ".npy" if upload_radar.name.endswith(".npy") else ".csv"
            )
        if mode_info.mode == "poster":
            st.sidebar.warning(
                "Poster Mode uses replayed predictions for analysis while showing your uploaded media."
            )

    base_df = preds_demo[preds_demo["scenario"] == live_scenario].copy()
    if base_df.empty:
        base_df = preds_demo[preds_demo["scenario"] == "clean"].copy()
        live_scenario = "clean"
    runtime_df = _build_runtime_predictions(
        base_df=base_df,
        audio_snr_db=audio_snr_db,
        vision_severity=vision_severity,
        radar_clutter_level=radar_clutter,
        miss_vis=miss_vis,
        miss_aud=miss_aud,
        miss_rad=miss_rad,
    )
    frame_count = len(runtime_df)
    selected_idx = int(st.sidebar.slider("Frame Index", 0, max(0, frame_count - 1), 0))
    current_row = runtime_df.iloc[selected_idx]
    overlay_frame, video_frames = _render_frame_with_bbox(
        video_path=video_path,
        bbox_df=bboxes,
        frame_idx=selected_idx,
        scenario=live_scenario,
    )

    current_controls = {
        "audio_snr_db": audio_snr_db,
        "vision_severity": vision_severity,
        "radar_clutter_level": radar_clutter,
        "miss_vis": miss_vis,
        "miss_aud": miss_aud,
        "miss_rad": miss_rad,
    }

    if st.session_state.judge_mode:
        steps = scripted_steps()
        idx = min(st.session_state.script_step_idx, len(steps) - 1)
        current = steps[idx]
        st.markdown("<div class='judge-title'>Demo Script</div>", unsafe_allow_html=True)
        checklist_cols = st.columns([5, 1.5, 1.5])
        labels = []
        for i, step in enumerate(steps):
            prefix = "[x]" if i <= idx else "[ ]"
            labels.append(f"{prefix} {step['name']}")
        checklist_cols[0].write(" -> ".join(labels))
        if checklist_cols[1].button("Start Demo", use_container_width=True):
            st.session_state.script_step_idx = 0
            st.rerun()
        if checklist_cols[2].button("Next", use_container_width=True):
            st.session_state.script_step_idx = min(idx + 1, len(steps) - 1)
            st.rerun()
        st.info(
            f"Current scripted step: {current['name']} | Scenario={current['scenario']} "
            f"| SNR={current['audio_snr_db']}dB | VisionSeverity={current['vision_severity']} "
            f"| RadarClutter={current['radar_clutter_level']}"
        )

    tabs = st.tabs(
        [
            "Live Inference",
            "One-click Judge Demo",
            "Robustness Lab",
            "Compare Models",
            "Latency / Deploy",
            "Export",
        ]
    )

    with tabs[0]:
        left, right = st.columns([3.2, 1.8])
        with left:
            st.subheader("Vision Panel")
            st.video(video_path.read_bytes())
            st.image(
                overlay_frame,
                caption=f"Overlay frame {selected_idx}/{max(0, video_frames-1)}",
                use_container_width=True,
            )
            if st.button("Take Screenshot for Poster", type="primary"):
                out_dir = _build_export_bundle(
                    mode=mode_info.mode,
                    scenario=live_scenario,
                    runtime_df=runtime_df,
                    frame_idx=selected_idx,
                    overlay_frame=overlay_frame,
                    controls=current_controls,
                )
                st.session_state.last_export_dir = str(out_dir)
                st.success(f"Saved poster screenshot bundle: {out_dir}")

            st.subheader("Audio Panel")
            waveform, sr = load_audio_waveform_from_path(str(audio_path))
            audio_fig = _plot_audio(waveform, sr)
            st.pyplot(audio_fig, use_container_width=True)
            plt.close(audio_fig)

            st.subheader("Radar Panel")
            if radar_path.suffix.lower() == ".npy":
                radar_matrix = np.load(str(radar_path))
                if radar_matrix.ndim == 1:
                    radar_matrix = radar_matrix.reshape(-1, 64)
            else:
                if radar_path == ASSETS_DIR / "sample_radar.csv":
                    radar_matrix = load_radar_matrix()
                else:
                    radar_df = pd.read_csv(radar_path)
                    radar_cols = [c for c in radar_df.columns if c.startswith("bin_")]
                    radar_matrix = radar_df[radar_cols].to_numpy(dtype=np.float32)
            radar_fig = _plot_radar(radar_matrix, selected_idx)
            st.pyplot(radar_fig, use_container_width=False)
            plt.close(radar_fig)

        with right:
            st.subheader("Decision Panel")
            st.metric("Fused Score (F-GATE)", f"{float(current_row['p_gate']):.3f}")
            st.metric("Vision", f"{float(current_row['p_vis']):.3f}")
            st.metric("Audio", f"{float(current_row['p_aud']):.3f}")
            st.metric("Radar", f"{float(current_row['p_rad']):.3f}")

            gating_fig = _plot_gating(runtime_df)
            st.pyplot(gating_fig, use_container_width=True)
            plt.close(gating_fig)

            conf_fig = _plot_confidence(current_row)
            st.pyplot(conf_fig, use_container_width=True)
            plt.close(conf_fig)

            st.markdown("**Quality Indicators**")
            st.write(
                {
                    "audio_snr_db": float(current_row["audio_snr_db"]),
                    "vision_degradation": float(current_row["vision_severity"]),
                    "radar_clutter_level": float(current_row["radar_clutter_level"]),
                    "missing_vision": int(current_row["miss_vis"] < 0.5),
                    "missing_audio": int(current_row["miss_aud"] < 0.5),
                    "missing_radar": int(current_row["miss_rad"] < 0.5),
                }
            )

    with tabs[1]:
        st.subheader("Scripted Judge Walkthrough")
        steps = scripted_steps()
        if st.button("Start Demo", key="judge_start"):
            st.session_state.script_step_idx = 0
        if st.button("Next Step", key="judge_next"):
            st.session_state.script_step_idx = min(
                st.session_state.script_step_idx + 1, len(steps) - 1
            )

        step_idx = st.session_state.script_step_idx
        step = steps[step_idx]
        st.markdown(
            f"**Step {step_idx + 1}/{len(steps)}: {step['name']}**  "
            f"Scenario `{step['scenario']}` | "
            f"SNR `{step['audio_snr_db']}` | Vision `{step['vision_severity']}` | "
            f"Radar `{step['radar_clutter_level']}`"
        )
        script_base = preds_demo[preds_demo["scenario"] == step["scenario"]].copy()
        if script_base.empty:
            script_base = preds_demo[preds_demo["scenario"] == "clean"].copy()
        script_runtime = _build_runtime_predictions(
            base_df=script_base,
            audio_snr_db=float(step["audio_snr_db"]),
            vision_severity=float(step["vision_severity"]),
            radar_clutter_level=float(step["radar_clutter_level"]),
            miss_vis=int(step["miss_vis"]),
            miss_aud=int(step["miss_aud"]),
            miss_rad=int(step["miss_rad"]),
        )
        script_row = script_runtime.iloc[min(selected_idx, len(script_runtime) - 1)]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vision", f"{float(script_row['p_vis']):.3f}")
        c2.metric("Audio", f"{float(script_row['p_aud']):.3f}")
        c3.metric("Radar", f"{float(script_row['p_rad']):.3f}")
        c4.metric("Fused (GATE)", f"{float(script_row['p_gate']):.3f}")
        fig = _plot_gating(script_runtime)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tabs[2]:
        st.subheader("Robustness Lab")
        col1, col2 = st.columns(2)
        with col1:
            lab_snr = st.slider("Lab Audio SNR", -10.0, 30.0, 10.0, 1.0)
            lab_vis = st.slider("Lab Vision Severity", 0.0, 3.0, 1.0, 0.25)
            lab_radar = st.slider("Lab Radar Clutter", 0.0, 1.0, 0.5, 0.05)
            lab_drop_vis = int(st.checkbox("Lab Drop Vision", value=False))
            lab_drop_aud = int(st.checkbox("Lab Drop Audio", value=False))
            lab_drop_rad = int(st.checkbox("Lab Drop Radar", value=False))

        lab_runtime = _build_runtime_predictions(
            base_df=base_df,
            audio_snr_db=lab_snr,
            vision_severity=lab_vis,
            radar_clutter_level=lab_radar,
            miss_vis=lab_drop_vis,
            miss_aud=lab_drop_aud,
            miss_rad=lab_drop_rad,
        )
        lab_row = lab_runtime.iloc[min(selected_idx, len(lab_runtime) - 1)]
        with col2:
            st.metric("Lab Fused (GATE)", f"{float(lab_row['p_gate']):.3f}")
            st.metric("Lab Vision", f"{float(lab_row['p_vis']):.3f}")
            st.metric("Lab Audio", f"{float(lab_row['p_aud']):.3f}")
            st.metric("Lab Radar", f"{float(lab_row['p_rad']):.3f}")
            lab_fig = _plot_gating(lab_runtime)
            st.pyplot(lab_fig, use_container_width=True)
            plt.close(lab_fig)

        st.markdown("**Expected Trends from Final Stress-Test CSVs**")
        stress_fig = _plot_stress_curves(
            tables["stress_audio"], tables["stress_vision"], tables["stress_radar"]
        )
        st.pyplot(stress_fig, use_container_width=True)
        plt.close(stress_fig)

    with tabs[3]:
        st.subheader("Compare Models")
        uni = tables["metrics_unimodal"].copy()
        fus = tables["metrics_fusion"].copy()
        fused_best = fus.sort_values("pr_auc", ascending=False).iloc[0]
        model_options = [
            "Vision-only",
            "Audio-only",
            "Radar-only",
            "F-AVG",
            "F-STATIC",
            "F-GATE",
        ]
        chosen = st.selectbox("Model", model_options, index=5)
        mapping = {
            "Vision-only": "UNI-V",
            "Audio-only": "UNI-A",
            "Radar-only": "UNI-R",
            "F-AVG": "F-AVG",
            "F-STATIC": "F-STATIC",
            "F-GATE": "F-GATE",
        }
        run_id = mapping[chosen]
        if run_id.startswith("UNI-"):
            row = uni[uni["run_id"] == run_id].iloc[0]
        else:
            row = fus[fus["run_id"] == run_id].iloc[0]

        a, b, c = st.columns(3)
        a.metric("PR-AUC", f"{float(row['pr_auc']):.4f}")
        b.metric("ROC-AUC", f"{float(row['roc_auc']):.4f}")
        c.metric("Best Fusion", f"{fused_best['run_id']} ({float(fused_best['pr_auc']):.4f})")

        comp = pd.concat(
            [
                uni[["run_id", "pr_auc", "roc_auc"]],
                fus[["run_id", "pr_auc", "roc_auc"]],
            ],
            ignore_index=True,
        )
        st.dataframe(comp.sort_values("pr_auc", ascending=False), use_container_width=True)

        fig, ax = plt.subplots(figsize=(7.8, 3.1))
        plot_df = comp.sort_values("pr_auc", ascending=False)
        ax.bar(plot_df["run_id"], plot_df["pr_auc"], color="#1f2937")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("PR-AUC")
        ax.set_title("Model Comparison")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tabs[4]:
        st.subheader("Latency / Deploy")
        latency = load_latency_summary()
        rows = []
        methods = latency.get("methods", {})
        for method_name, values in methods.items():
            wall = values.get("wall_clock", {})
            rows.append(
                {
                    "method": method_name,
                    "mean_ms": float(wall.get("mean_ms", 0.0)),
                    "p95_ms": float(wall.get("p95_ms", 0.0)),
                    "n": int(wall.get("n", 0)),
                }
            )
        latency_df = pd.DataFrame(rows)
        st.dataframe(latency_df, use_container_width=True)
        fig, ax = plt.subplots(figsize=(7.2, 3.1))
        ax.bar(latency_df["method"], latency_df["mean_ms"], color="#0f766e")
        ax.set_ylabel("Mean Latency (ms)")
        ax.set_title("Fusion Latency (Wall Clock)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if st.button("Measure current replay latency (200 frames)"):
            start = time.perf_counter()
            _ = _build_runtime_predictions(
                base_df=base_df,
                audio_snr_db=audio_snr_db,
                vision_severity=vision_severity,
                radar_clutter_level=radar_clutter,
                miss_vis=miss_vis,
                miss_aud=miss_aud,
                miss_rad=miss_rad,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            st.success(f"Replay inference path latency: {elapsed_ms:.3f} ms")

    with tabs[5]:
        st.subheader("Export")
        st.write("Generate a poster-ready export bundle from current state.")
        if st.button("Export poster bundle", type="primary"):
            out_dir = _build_export_bundle(
                mode=mode_info.mode,
                scenario=live_scenario,
                runtime_df=runtime_df,
                frame_idx=selected_idx,
                overlay_frame=overlay_frame,
                controls=current_controls,
            )
            st.session_state.last_export_dir = str(out_dir)
            st.success(f"Exported bundle to: {out_dir}")
        if st.session_state.last_export_dir:
            st.info(f"Last export: {st.session_state.last_export_dir}")
        st.code(
            "Saved files: overlay_frame.png, gating_weights_chart.png, "
            "confidence_panel.png, gating_weights.csv, predictions_per_frame.csv, "
            "run_settings.json, caption.txt"
        )

    footer = st.container()
    with footer:
        st.caption(
            "Sample metadata: "
            f"source_video={metadata.get('source_video')} | "
            f"source_audio={metadata.get('source_audio')} | "
            f"frame_step={metadata.get('frame_step')} | "
            f"mode={mode_info.mode}"
        )


if __name__ == "__main__":
    main()
