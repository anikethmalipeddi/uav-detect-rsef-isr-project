# Results Section Asset Manifest

This folder contains poster-ready RESULTS assets for ISEF science fair layout.
All metrics are TEST split unless explicitly stated.

## Asset Index
### Table R1: `tables/R1_overall_test_performance.png (+ .csv)`
- **What it shows:** Overall TEST comparison of all unimodal and fusion variants with thresholded operating metrics.
- **Data sources:** results/metrics_unimodal.csv, results/metrics_fusion.csv, results/preds_fusion.csv
- **Poster caption (paste as-is):** Table R1. Episode-level TEST performance across unimodal and fusion variants at a fixed threshold of 0.5 (FAR = FP/(FP+TN)).

### Figure R1: `figures/R1_pr_curves_test.png`
- **What it shows:** Precision-Recall overlay for UNI-A plus fusion baselines and adaptive variants.
- **Data sources:** results/preds_fusion.csv
- **Poster caption (paste as-is):** Figure R1. TEST precision-recall curves for UNI-A, F-AVG, F-STATIC, F-GATE, and F-GATE-NOQ.

### Figure R2: `figures/R2_roc_curves_test.png`
- **What it shows:** ROC overlay for the same model set as Figure R1.
- **Data sources:** results/preds_fusion.csv
- **Poster caption (paste as-is):** Figure R2. TEST ROC curves for the same models as Figure R1, using identical episode-level predictions.

### Figure R3: `figures/R3_confusion_matrices_binary_test.png`
- **What it shows:** 2x4 panel of binary episode-level confusion matrices for all 7 models at threshold 0.5.
- **Data sources:** results/preds_fusion.csv
- **Poster caption (paste as-is):** Figure R3. Episode-level binary confusion matrices (threshold = 0.5) for UNI-V, UNI-A, UNI-R, F-AVG, F-STATIC, F-GATE, and F-GATE-NOQ.

### Figure R4: `figures/R4_missing_modality_robustness.png`
- **What it shows:** PR-AUC vs missing-modality scenario comparing UNI-best-available, F-AVG, and F-GATE.
- **Data sources:** results/missing_modality.csv, results/metrics_unimodal.csv
- **Poster caption (paste as-is):** Figure R4. Missing-modality robustness on TEST across all_present, single-missing, and single-modality scenarios.

### Table R2: `tables/R2_missing_modality_per_scenario.png (+ .csv)`
- **What it shows:** Per-scenario PR-AUC and FAR for UNI-best-available, F-AVG, and F-GATE.
- **Data sources:** results/missing_modality.csv, results/metrics_unimodal.csv
- **Poster caption (paste as-is):** Table R2. Per-scenario missing-modality performance; UNI-best-available selects the strongest available unimodal branch for each scenario.

### Figure R5: `figures/R5_gate_vs_noq_delta.png`
- **What it shows:** Scenario-wise delta PR-AUC between F-GATE and F-GATE-NOQ (overall TEST + missingness scenarios).
- **Data sources:** results/metrics_fusion.csv, results/missing_modality.csv
- **Poster caption (paste as-is):** Figure R5. Effect of quality signals: PR-AUC difference (F-GATE minus F-GATE-NOQ) over TEST and missing-modality scenarios.

### Table R3: `tables/R3_ablation_deltas_gate_vs_noq.png (+ .csv)`
- **What it shows:** Overall TEST deltas for F-GATE minus F-GATE-NOQ on PR-AUC, AUROC, Accuracy, and FAR.
- **Data sources:** results/metrics_fusion.csv
- **Poster caption (paste as-is):** Table R3. Overall TEST deltas for F-GATE relative to F-GATE-NOQ at threshold 0.5.

### Figure R6a: `figures/R6a_stress_audio_snr.png`
- **What it shows:** F-GATE PR-AUC trend over audio SNR with unimodal TEST baselines as references.
- **Data sources:** results/stress_audio_snr.csv, results/metrics_unimodal.csv
- **Poster caption (paste as-is):** Figure R6a. Audio SNR stress sweep (TEST): F-GATE trend with unimodal TEST baselines shown as dashed references.

### Figure R6b: `figures/R6b_stress_vision_corruption.png`
- **What it shows:** F-GATE PR-AUC trend over vision corruption severity with unimodal TEST baseline references.
- **Data sources:** results/stress_vision_corruptions.csv, results/metrics_unimodal.csv
- **Poster caption (paste as-is):** Figure R6b. Vision corruption stress sweep (TEST): F-GATE trend with unimodal TEST baselines shown as dashed references.

### Figure R6c: `figures/R6c_stress_radar_clutter.png`
- **What it shows:** F-GATE PR-AUC trend over radar clutter level with unimodal TEST baseline references.
- **Data sources:** results/stress_radar_clutter.csv, results/metrics_unimodal.csv
- **Poster caption (paste as-is):** Figure R6c. Radar clutter stress sweep (TEST): F-GATE trend with unimodal TEST baselines shown as dashed references.

### Table R4: `tables/R4_dataset_split_counts.png (+ .csv)`
- **What it shows:** Train/val/test episode counts and class balance.
- **Data sources:** data/processed/episodes/episodes.csv
- **Poster caption (paste as-is):** Table R4. Dataset split composition and class balance for train, validation, and held-out TEST episodes.

### Figure R7: `figures/R7_latency_summary.png`
- **What it shows:** Fusion latency summary callout (reused curated artifact when available).
- **Data sources:** final_isef_package/figures/latency_breakdown.png, results/latency_summary.json
- **Poster caption (paste as-is):** Figure R7. Runtime summary of fusion methods (mean latency, with P95 context from the measured latency report).
