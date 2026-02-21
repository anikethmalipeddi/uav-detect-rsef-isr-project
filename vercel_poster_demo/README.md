# ISEF Demo (Next.js)

This folder contains the visual-first judge demo for multimodal UAV detection using precomputed outputs from the final run. The app is designed for a 60-120 second walkthrough showing: detector operation, fusion-vs-unimodal behavior on hard cases, and adaptive fusion weighting under degradation/missingness.

## Running the Demo

1. `cd vercel_poster_demo`
2. `npm install`
3. `npm run dev`
4. Open `http://localhost:3000`

## Production Build

1. `cd vercel_poster_demo`
2. `npm install`
3. `npm run build`
4. `npm run start`

## Vercel Deployment

- Set **Root Directory** to `vercel_poster_demo`.
- Framework preset: Next.js.

## Adding a Scenario

1. Add scenario outputs to `public/demo/scenarios.json`.
2. Add media to `public/demo/videos/` if needed.
3. Include required fields:
   - `id`, `title`, `demonstrates`, `wow`, `wow_tag`
   - `ground_truth`
   - `media.video`
   - `quality` flags (`good|degraded|missing`) for `vision`, `audio`, `radar`
   - `coordinate_type`, `source_resolution`
   - `frames` array with per-frame detections/scores/weights
4. Keep placeholders out of the default scenario list (`meta.default_scenarios`).

## Data Format

- Scenario source: `public/demo/scenarios.json`
- Coordinate modes:
  - `coordinate_type: "pixel"` means boxes are in source pixel coordinates and scaled by `source_resolution`
  - `coordinate_type: "normalized"` means boxes are in `[0,1]`
- Per-frame schema:

```json
{
  "frame_idx": 42,
  "t": 5.25,
  "vision": { "boxes": [{ "x": 80, "y": 40, "w": 60, "h": 30, "class": "DRONE", "conf": 0.61 }], "score": 0.61 },
  "fusion": { "boxes": [{ "x": 82, "y": 41, "w": 60, "h": 30, "class": "DRONE", "conf": 0.67 }], "score": 0.67 },
  "modality_scores": { "vision": 0.61, "audio": 0.55, "radar": 0.52, "fusion": 0.67 },
  "weights": { "vision": 0.34, "audio": 0.38, "radar": 0.28 }
}
```

## Main Files

- `src/app/page.tsx` - single-page ISEF demo
- `src/components/VideoOverlayPanel.tsx` - video + canvas overlay renderer
- `src/components/ScenarioSelector.tsx` - scenario selector pills
- `src/components/AudioWaveform.tsx` - audio visual sample
- `src/components/RadarSnapshot.tsx` - radar visual sample
- `public/demo/scenarios.json` - curated scenario set (includes WOW cases)
