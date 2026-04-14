# Sources And Notes

This note records the exact repo files used to build each presentation asset and the selection choices made during generation.

## Primary final metrics sources

- `final_isef_package/results_section/tables/R1_overall_test_performance.csv`
  Used for exact TEST PR-AUC and exact TEST false alarm rate values in Figures 05 and 06.
- `final_isef_package/metadata/significance_pr_auc.txt`
  Used for the PR-AUC callout in Figure 05, specifically the reported `F-GATE - F-AVG` mean difference and 95% CI.
- `final_isef_package/tables/stress_radar_clutter.csv`
  Used for the radar clutter stress inset in Figure 06.

## Selected project-derived raw inputs

### Vision challenge selection

- `final_isef_package/verification/IR_DRONE_059_f000112_DRONE.png`
  Chosen as the drone example because it is a real project verification frame with a distant IR target and limited detail.
- `final_isef_package/verification/IR_BIRD_010_f000043_BIRD.png`
  Chosen as the confusing negative because it provides a real bird silhouette under similar distant, low-detail conditions.

Reason for selection:
These two verification images most directly communicate the class-boundary ambiguity described in the project without introducing decorative or off-project imagery.

### Audio challenge selection

- `data/processed/audio_windows/DRONE_020_00000.npy`
- `data/processed/episodes/episodes.csv`

Chosen episode metadata:

- `episode_id=003731`
- `source_audio_label=Drone`
- `audio_snr_db=-25.1956`

Reason for selection:
This processed log-mel window preserved visible harmonic structure while still showing a strong low-SNR noise context, making it interpretable on a slide and faithful to the project’s acoustic challenge.

### Radar challenge selection

- `data/processed/radar_rd_real/rd_00047495.npy`
- `data/processed/episodes/episodes.csv`

Chosen episode metadata:

- `episode_id=003737`
- `source_radar_label=Drone`
- `radar_clutter_level=4.0452`

Reason for selection:
This range-Doppler sample showed both a visible target return and meaningful interference/clutter structure, which made it better for annotation than cleaner but less informative alternatives.

## Asset-by-asset source map

### 01_system_overview

Sources:

- `final_isef_package/README.md`
- `final_isef_package/methodology_materials_updated.md`

Notes:
Author-created overview diagram. Terminology was kept consistent with the final project wording: vision, acoustic, radar, adaptive fusion, and final UAV / Not UAV decision.

### 02a_vision_challenge

Sources:

- `final_isef_package/verification/IR_DRONE_059_f000112_DRONE.png`
- `final_isef_package/verification/IR_BIRD_010_f000043_BIRD.png`

Notes:
Fully project-derived panel using real verification imagery. Labels were kept descriptive and honest to the visible evidence rather than forcing morphology claims that were not clearly visible.

### 02b_audio_challenge

Sources:

- `data/processed/audio_windows/DRONE_020_00000.npy`
- `data/processed/episodes/episodes.csv` for provenance

Notes:
Fully project-derived panel. The plotted view is a cleaned presentation rendering of the actual processed spectrogram window, not a fabricated schematic.

### 02c_radar_challenge

Sources:

- `data/processed/radar_rd_real/rd_00047495.npy`
- `data/processed/episodes/episodes.csv` for provenance

Notes:
Fully project-derived panel. The plotted heatmap is a cleaned presentation rendering of the actual processed range-Doppler sample.

### 02_modality_challenges_combined

Sources:

- `02a_vision_challenge`
- `02b_audio_challenge`
- `02c_radar_challenge`

Notes:
Composite figure assembled from the three finalized modality challenge panels so the design language stays consistent across the combined slide version and the standalone exports.

### 03_procedure_pipeline

Sources:

- `final_isef_package/methodology_materials_updated.md`

Notes:
Author-created workflow diagram using the final project sequence:
Prepare data; train modality models; extract confidence and quality signals; compare fusion methods; evaluate under normal and stressed conditions.

### 04_system_architecture

Sources:

- `final_isef_package/methodology_materials_updated.md`
- `final_isef_package/methodology_assets/tables/method_table_M2.csv`
- `final_isef_package/methodology_assets/tables/method_table_M3.csv`
- `selected_project_inputs/vision_drone_IR_DRONE_059_f000112_DRONE.png`
- `selected_project_inputs/audio_DRONE_020_00000_preview.png`
- `selected_project_inputs/radar_rd_00047495_preview.png`

Notes:
Mixed figure. The branch structure and terminology are author-created, but the left-side modality thumbnails are drawn from real project inputs.

### 05_prauc_results

Sources:

- `final_isef_package/results_section/tables/R1_overall_test_performance.csv`
- `final_isef_package/metadata/significance_pr_auc.txt`

Exact values used:

- `UNI-A` PR-AUC `0.7443546231`
- `F-AVG` PR-AUC `0.8505706106`
- `F-STATIC` PR-AUC `0.8845568501`
- `F-GATE` PR-AUC `0.8905556586`
- `F-GATE-NOQ` PR-AUC `0.8941218958`

Notes:

- Figure 05 includes `F-GATE-NOQ` because it has the highest exact overall TEST PR-AUC in the final performance table.
- The callout box uses the significance file’s reported `F-GATE - F-AVG` mean difference `0.039713` and 95% CI `(0.003825, 0.071903)`.
- The best unimodal baseline shown is `UNI-A`, because it is the strongest single-modality TEST PR-AUC in the final table.

### 06_false_alarm_and_stress

Sources:

- `final_isef_package/results_section/tables/R1_overall_test_performance.csv`
- `final_isef_package/tables/stress_radar_clutter.csv`
- `data/processed/radar_rd_real/rd_00047495.npy`

Exact false alarm values used:

- `UNI-V` FAR `0.745`
- `F-AVG` FAR `0.385`
- `F-STATIC` FAR `0.205`
- `F-GATE` FAR `0.210`
- `F-GATE-NOQ` FAR `0.220`

Exact radar clutter inset values used:

- `F-GATE` PR-AUC at clutter `0.1`: `0.8952481554`
- `F-GATE` PR-AUC at clutter `0.9`: `0.8823662735`

Notes:

- Figure 06 uses `F-GATE` for the stress inset because the radar clutter stress table is explicitly reported for `F-GATE`, and the significance metadata is also centered on `F-GATE` comparisons.
- The inset is therefore aligned to the quality-aware adaptive fusion result emphasized in the project’s robustness discussion.

### 07_conclusion_summary_visual

Sources:

- `final_isef_package/results_section/tables/R1_overall_test_performance.csv`
- `final_isef_package/tables/stress_radar_clutter.csv`

Notes:
Author-created synthesis visual. The three statements were constrained to findings supported by the final metrics: stronger detection, better robustness under degraded inputs, and lower false alarms for practical use.

### 08_future_work_roadmap

Sources:

- `final_isef_package/methodology_materials_updated.md`
- `final_isef_package/README.md`

Notes:
Author-created roadmap using final project-consistent next steps rather than adding speculative directions not present in the repo materials.

## Terminology alignment notes

- Used `Acoustic` in the high-level overview to match the requested presentation framing, while keeping model-specific wording tied to spectrogram features in the architecture figure.
- Used `Adaptive fusion` and `Adaptive gating` in places where the project emphasizes quality-aware fusion behavior.
- Used `UAV / Not UAV` for the final decision label because it is concise for slides while still matching the project detection framing.

## Schematic vs. project-derived usage

- Purely author-created diagrams: `01_system_overview`, `03_procedure_pipeline`, `07_conclusion_summary_visual`, `08_future_work_roadmap`
- Mixed diagram with project-derived thumbnails: `04_system_architecture`
- Fully project-derived figures or panels: `02a_vision_challenge`, `02b_audio_challenge`, `02c_radar_challenge`, `02_modality_challenges_combined`, `05_prauc_results`, `06_false_alarm_and_stress`

## Manual review status

No unresolved metric conflict remained after checking the final package tables and metadata, so `review_flags.md` was not created.
