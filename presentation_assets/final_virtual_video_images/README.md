# Presentation figures

PNG and SVG figures used in the ISR/RSEF presentation. The table maps each export to its source and suggested placement.

## Where to use each asset

| File | Suggested slide use | Basis |
| --- | --- | --- |
| `final_png/01_system_overview.png` and `final_svg/01_system_overview.svg` | Opening project overview | Author-created system diagram aligned to final project terminology |
| `final_png/02_modality_challenges_combined.png` and `final_svg/02_modality_challenges_combined.svg` | Why fusion matters / modality limitations | Composite built from real project-derived vision, audio, and radar examples |
| `final_png/02a_vision_challenge.png` and `final_svg/02a_vision_challenge.svg` | Optional standalone vision challenge slide or animation build | Real project verification images |
| `final_png/02b_audio_challenge.png` and `final_svg/02b_audio_challenge.svg` | Optional standalone audio challenge slide or animation build | Real project spectrogram window |
| `final_png/02c_radar_challenge.png` and `final_svg/02c_radar_challenge.svg` | Optional standalone radar challenge slide or animation build | Real project range-Doppler sample |
| `final_png/03_procedure_pipeline.png` and `final_svg/03_procedure_pipeline.svg` | High-level procedure / methods workflow | Author-created workflow diagram from final methodology wording |
| `final_png/04_system_architecture.png` and `final_svg/04_system_architecture.svg` | Technical architecture slide | Mixed diagram with project-derived sample thumbnails and method labels from final materials |
| `final_png/05_prauc_results.png` and `final_svg/05_prauc_results.svg` | Main results slide | Exact chart from final project metrics |
| `final_png/06_false_alarm_and_stress.png` and `final_svg/06_false_alarm_and_stress.svg` | Reliability / stress-testing slide | Exact false alarm metrics plus exact radar-clutter stress result |
| `final_png/07_conclusion_summary_visual.png` and `final_svg/07_conclusion_summary_visual.svg` | Conclusion slide | Author-created synthesis visual grounded in final findings |
| `final_png/08_future_work_roadmap.png` and `final_svg/08_future_work_roadmap.svg` | Future work slide | Author-created roadmap using final project direction statements |
| `contact_sheet.png` | Quick visual review of the full set | Thumbnail overview of all final assets |

## Real project-derived inputs

Selected raw examples copied for traceability are in `selected_project_inputs/`.

- `vision_drone_IR_DRONE_059_f000112_DRONE.png`
- `vision_bird_IR_BIRD_010_f000043_BIRD.png`
- `audio_DRONE_020_00000.npy` and `audio_DRONE_020_00000_preview.png`
- `radar_rd_00047495.npy` and `radar_rd_00047495_preview.png`

## Supporting files

- `supporting_files/generate_presentation_assets.py`: reproducible asset-generation script
- `supporting_files/asset_manifest.json`: compact asset/source manifest
- `source_work/`: intermediate candidate sheets and working previews used during selection
