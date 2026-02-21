# Vercel Poster Mode Demo

This directory contains a static Poster Mode demo that runs entirely from precomputed JSON and small image assets.

## Local test

From repo root:

```powershell
python -m http.server 8080
```

Then open:

`http://localhost:8080/vercel_poster_demo/`

## Deploy to Vercel

`vercel.json` in repo root routes all requests to this directory.

### CLI

```bash
npm i -g vercel
vercel --prod
```

### GitHub integration

1. Import this repo in Vercel.
2. Keep default framework detection (Other).
3. Deploy.

## Refresh data/assets

When final package tables are updated, regenerate this app data:

```powershell
C:\Users\Aniketh\miniforge3\envs\uavmm\python.exe tools/build_vercel_poster_assets.py
```

