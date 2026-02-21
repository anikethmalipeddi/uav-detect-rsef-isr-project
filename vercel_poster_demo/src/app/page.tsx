"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import AudioWaveform from "@/components/AudioWaveform";
import RadarSnapshot from "@/components/RadarSnapshot";
import ScenarioSelector from "@/components/ScenarioSelector";
import StatusChip from "@/components/StatusChip";
import VideoOverlayPanel from "@/components/VideoOverlayPanel";
import { classEntries, toClassDistribution, topClassFromDroneProb } from "@/lib/classUtils";
import { AudioSample, DemoData, Downloadable, FrameRow, RadarRow, Scenario } from "@/lib/types";

const DEFAULT_PERFORMANCE_STEP = 1;

type FusionMetricRow = {
  run_id: string;
  split: string;
  pr_auc: number;
  roc_auc: number;
};
type UnimodalMetricRow = {
  run_id: string;
  split: string;
  pr_auc: number;
  roc_auc: number;
};

function validateDemoData(payload: unknown): payload is DemoData {
  if (!payload || typeof payload !== "object") return false;
  const obj = payload as Record<string, unknown>;
  if (!Array.isArray(obj.scenarios)) return false;
  if (!obj.meta || typeof obj.meta !== "object") return false;
  const scenarios = obj.scenarios as unknown[];
  if (scenarios.length < 5) return false;
  return scenarios.every((s) => {
    if (!s || typeof s !== "object") return false;
    const r = s as Record<string, unknown>;
    return typeof r.id === "string" && typeof r.title === "string" && Array.isArray(r.frames) && r.frames.length > 0;
  });
}

function nearestFrame(frames: FrameRow[], tSec: number): FrameRow {
  if (!frames.length) {
    throw new Error("Scenario has no frame data.");
  }
  if (frames.length === 1) return frames[0];
  const firstT = frames[0].t;
  const lastT = frames[frames.length - 1].t;
  if (tSec <= firstT) return frames[0];
  if (tSec >= lastT) return frames[frames.length - 1];

  const fps = (frames.length - 1) / Math.max(1e-6, lastT - firstT);
  const idx = Math.max(0, Math.min(frames.length - 1, Math.round((tSec - firstT) * fps)));
  return frames[idx];
}

function fmtPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function getScenarioVideoFilter(scenarioId: string): string {
  switch (scenarioId) {
    case "vision_degraded":
      return "brightness(0.58) contrast(1.12) blur(1.4px)";
    case "audio_degraded":
      return "none";
    case "radar_clutter":
      return "contrast(1.06)";
    case "missing_vision":
      return "brightness(0.42) grayscale(1) blur(2.2px)";
    case "missing_radar":
      return "none";
    case "missing_audio":
      return "none";
    default:
      return "none";
  }
}

export default function Page() {
  const [demo, setDemo] = useState<DemoData | null>(null);
  const [audio, setAudio] = useState<AudioSample[]>([]);
  const [radar, setRadar] = useState<RadarRow[]>([]);
  const [downloadables, setDownloadables] = useState<Downloadable[]>([]);
  const [fusionMetrics, setFusionMetrics] = useState<FusionMetricRow[]>([]);
  const [unimodalMetrics, setUnimodalMetrics] = useState<UnimodalMetricRow[]>([]);

  const [error, setError] = useState<string>("");
  const [scenarioId, setScenarioId] = useState<string>("");
  const [posterMode, setPosterMode] = useState(false);
  const [showScript, setShowScript] = useState(false);
  const [showComparison, setShowComparison] = useState(true);
  const [showDownloadables, setShowDownloadables] = useState(false);
  const [performanceStep, setPerformanceStep] = useState(DEFAULT_PERFORMANCE_STEP);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(15);

  const visionVideoRef = useRef<HTMLVideoElement>(null);
  const fusionVideoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [scenarioRes, audioRes, radarRes, downloadsRes, fusionMetricsRes, unimodalMetricsRes] = await Promise.all([
          fetch("/demo/scenarios.json", { cache: "no-store" }),
          fetch("/demo/data/audio_features.json", { cache: "force-cache" }),
          fetch("/demo/data/radar_features.json", { cache: "force-cache" }),
          fetch("/demo/downloadables/manifest.json", { cache: "no-store" }),
          fetch("/demo/data/metrics_fusion.json", { cache: "force-cache" }),
          fetch("/demo/data/metrics_unimodal.json", { cache: "force-cache" }),
        ]);

        if (!scenarioRes.ok) throw new Error(`Failed to load scenarios.json (${scenarioRes.status}).`);
        const scenarioPayload = (await scenarioRes.json()) as unknown;
        if (!validateDemoData(scenarioPayload)) {
          throw new Error("Scenario JSON is malformed. Check required fields in /public/demo/scenarios.json.");
        }
        const parsedDemo = scenarioPayload as DemoData;

        const parsedAudio = audioRes.ok ? ((await audioRes.json()) as AudioSample[]) : [];
        const parsedRadar = radarRes.ok ? ((await radarRes.json()) as RadarRow[]) : [];
        const parsedDownloads = downloadsRes.ok ? ((await downloadsRes.json()) as Downloadable[]) : [];
        const parsedFusionMetrics = fusionMetricsRes.ok ? ((await fusionMetricsRes.json()) as FusionMetricRow[]) : [];
        const parsedUnimodalMetrics = unimodalMetricsRes.ok ? ((await unimodalMetricsRes.json()) as UnimodalMetricRow[]) : [];

        setDemo(parsedDemo);
        setAudio(Array.isArray(parsedAudio) ? parsedAudio : []);
        setRadar(Array.isArray(parsedRadar) ? parsedRadar : []);
        setDownloadables(Array.isArray(parsedDownloads) ? parsedDownloads : []);
        setFusionMetrics(Array.isArray(parsedFusionMetrics) ? parsedFusionMetrics : []);
        setUnimodalMetrics(Array.isArray(parsedUnimodalMetrics) ? parsedUnimodalMetrics : []);

        const defaultScenario = parsedDemo.meta.default_scenarios?.[0] || parsedDemo.scenarios[0]?.id || "";
        setScenarioId(defaultScenario);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load demo data.");
      }
    };

    load();
  }, []);

  const scenarios = demo?.scenarios ?? [];

  const scenario = useMemo<Scenario | null>(() => {
    if (!scenarios.length) return null;
    return scenarios.find((s) => s.id === scenarioId) ?? scenarios[0];
  }, [scenarios, scenarioId]);

  const frame = useMemo<FrameRow | null>(() => {
    if (!scenario) return null;
    return nearestFrame(scenario.frames, currentTime);
  }, [scenario, currentTime]);

  const frameIdx = frame?.frame_idx ?? 0;
  const visionAvailable = scenario?.quality.vision !== "missing";
  const showVisionPanel = !posterMode && visionAvailable;
  const scenarioVideoFilter = getScenarioVideoFilter(scenario?.id ?? "clean");
  const fusionContextOnly = scenario?.quality.vision === "missing";

  useEffect(() => {
    if (!scenario) return;
    setDuration(Math.max(1, scenario.frames[scenario.frames.length - 1]?.t || 15));
    const focus = scenario.frames.find((f) => f.frame_idx === scenario.focus_frame);
    const startT = focus?.t ?? 0;
    setCurrentTime(startT);
    setIsPlaying(false);

    const vv = visionVideoRef.current;
    const fv = fusionVideoRef.current;
    if (vv) {
      vv.pause();
      vv.currentTime = startT;
    }
    if (fv) {
      fv.pause();
      fv.currentTime = startT;
    }
  }, [scenario?.id]);

  useEffect(() => {
    if (!isPlaying) return;
    const timer = window.setInterval(() => {
      const master = fusionVideoRef.current;
      const follower = visionVideoRef.current;
      if (!master) return;

      const mt = master.currentTime;
      setCurrentTime(mt);

      if (follower) {
        const drift = Math.abs(follower.currentTime - mt);
        if (drift > 0.06) {
          follower.currentTime = mt;
        }
      }

      if (master.ended) {
        setIsPlaying(false);
      }
    }, 80);

    return () => window.clearInterval(timer);
  }, [isPlaying]);

  useEffect(() => {
    if (!posterMode) {
      setShowComparison(true);
    }
  }, [posterMode]);

  const onPlayPause = async () => {
    const videos = [fusionVideoRef.current, visionVideoRef.current].filter(Boolean) as HTMLVideoElement[];
    if (!videos.length) return;

    if (isPlaying) {
      videos.forEach((v) => v.pause());
      setIsPlaying(false);
      return;
    }

    await Promise.all(videos.map((v) => v.play().catch(() => undefined)));
    setIsPlaying(true);
  };

  const onScrub = (value: number) => {
    setCurrentTime(value);
    const videos = [fusionVideoRef.current, visionVideoRef.current].filter(Boolean) as HTMLVideoElement[];
    videos.forEach((v) => {
      v.currentTime = value;
    });
  };

  const onReset = () => {
    if (!demo) return;
    setScenarioId(demo.meta.default_scenarios?.[0] ?? demo.scenarios[0].id);
    setPosterMode(false);
    setShowScript(false);
    setShowComparison(true);
    setShowDownloadables(false);
    setPerformanceStep(DEFAULT_PERFORMANCE_STEP);
    onScrub(0);
    const videos = [fusionVideoRef.current, visionVideoRef.current].filter(Boolean) as HTMLVideoElement[];
    videos.forEach((v) => v.pause());
    setIsPlaying(false);
  };

  const gt = scenario?.ground_truth.class ?? "DRONE";

  const visionTop = frame ? topClassFromDroneProb(frame.modality_scores.vision) : { label: gt, conf: 0 };
  const audioTop = frame ? topClassFromDroneProb(frame.modality_scores.audio) : { label: gt, conf: 0 };
  const radarTop = frame ? topClassFromDroneProb(frame.modality_scores.radar) : { label: gt, conf: 0 };
  const fusionTop = frame ? topClassFromDroneProb(frame.modality_scores.fusion) : { label: gt, conf: 0 };

  const fusionDist = frame ? toClassDistribution(frame.modality_scores.fusion) : toClassDistribution(0.5);
  const fusionDistEntries = classEntries(fusionDist);

  const delta = frame ? frame.modality_scores.fusion - frame.modality_scores.vision : 0;

  const deltaMessage = (() => {
    if (scenario?.quality.vision === "missing") return "Vision is missing in this scenario. Fusion reweights to audio/radar.";
    if (scenario?.quality.audio === "missing") return "Audio is missing in this scenario. Fusion reweights to vision/radar.";
    if (scenario?.quality.radar === "missing") return "Radar is missing in this scenario. Fusion reweights to vision/audio.";
    if (visionTop.label !== gt && fusionTop.label === gt) return "Fusion correct while Vision-only is wrong.";
    if (visionTop.label === gt && fusionTop.label === gt && delta >= 0) {
      return `Both correct; fusion margin +${fmtPct(Math.max(0, delta))}.`;
    }
    if (fusionTop.label === gt) return "Fusion remains correct under stress.";
    return "Hard case across models.";
  })();

  const currentRadarRow = useMemo(() => {
    if (!radar.length || !frame) return null;
    const row = radar.find((r) => Number(r.frame_idx) === frame.frame_idx);
    return row ?? radar[0] ?? null;
  }, [radar, frame]);

  const testFusionMetrics = useMemo(
    () => fusionMetrics.filter((m) => m.split.toLowerCase() === "test"),
    [fusionMetrics],
  );
  const bestFusionRunId = useMemo(() => {
    if (!testFusionMetrics.length) return "F-GATE-NOQ";
    return [...testFusionMetrics].sort((a, b) => b.pr_auc - a.pr_auc)[0].run_id;
  }, [testFusionMetrics]);
  const bestFusionMetric = useMemo(() => {
    if (!testFusionMetrics.length) return null;
    return [...testFusionMetrics].sort((a, b) => b.pr_auc - a.pr_auc)[0];
  }, [testFusionMetrics]);
  const visionUnimodalMetric = useMemo(() => {
    const testUni = unimodalMetrics.filter((m) => m.split.toLowerCase() === "test");
    return testUni.find((m) => m.run_id === "UNI-V") ?? null;
  }, [unimodalMetrics]);

  const ablationFigure = useMemo(() => {
    const row = downloadables.find((d) => d.name === "ablation_bars.png");
    return row?.path || null;
  }, [downloadables]);
  const leaderboardRows = useMemo(() => {
    const combined = [...testFusionMetrics, ...unimodalMetrics.filter((m) => m.split.toLowerCase() === "test")];
    return combined.sort((a, b) => b.pr_auc - a.pr_auc);
  }, [testFusionMetrics, unimodalMetrics]);
  const leaderboardMaxPrAuc = useMemo(() => {
    if (!leaderboardRows.length) return 1;
    return Math.max(...leaderboardRows.map((m) => m.pr_auc));
  }, [leaderboardRows]);

  const signalLabel = (score: number, missing: boolean): string => {
    if (missing) return "N/A";
    if (score >= 0.6) return "Strong";
    if (score >= 0.52) return "Moderate";
    if (score >= 0.45) return "Uncertain";
    return "Weak";
  };

  if (error) {
    return (
      <main className="mx-auto max-w-6xl p-6">
        <div className="rounded-2xl border border-rose-400 bg-rose-50 p-5 text-rose-900">
          <h1 className="mb-2 text-2xl font-bold">Demo Data Error</h1>
          <p>{error}</p>
          <p className="mt-3 text-sm">Fix `/public/demo/scenarios.json` and refresh.</p>
        </div>
      </main>
    );
  }

  if (!demo || !scenario || !frame) {
    return (
      <main className="mx-auto max-w-6xl p-6">
        <div className="rounded-2xl border border-slate-300 bg-white p-5 text-slate-700">Loading demo assets...</div>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-[1800px] p-3 md:p-4">
      <header className="mb-3 rounded-2xl border border-slate-300 bg-white p-3 shadow-sm md:p-4">
        <div className="grid gap-3 md:grid-cols-[1fr_2fr_1fr] md:items-center">
          <div>
            <h1 className={`${posterMode ? "text-3xl" : "text-2xl"} font-black leading-tight text-slate-900`}>
              UAV Multimodal Fusion Demo
            </h1>
          </div>

          <ScenarioSelector
            scenarios={scenarios}
            selectedId={scenario.id}
            onSelect={(id) => setScenarioId(id)}
            posterMode={posterMode}
          />

          <div className="flex flex-wrap items-center justify-end gap-2">
            <button
              onClick={() => setPosterMode((v) => !v)}
              className={`rounded-lg border px-3 py-1.5 font-semibold ${posterMode ? "border-cyan-700 bg-cyan-700 text-white" : "border-slate-400 bg-white text-slate-900"}`}
            >
              Poster Mode {posterMode ? "On" : "Off"}
            </button>
            <button onClick={onReset} className="rounded-lg border border-slate-400 bg-white px-3 py-1.5 font-semibold text-slate-900">
              Reset to default
            </button>
            <button
              onClick={() => setShowScript(true)}
              className="rounded-lg border border-indigo-700 bg-indigo-700 px-3 py-1.5 font-semibold text-white"
            >
              Demo script
            </button>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
          <span className="rounded-full bg-slate-100 px-3 py-1 font-semibold text-slate-800">{scenario.demonstrates}</span>
          {scenario.wow ? <span className="rounded-full bg-fuchsia-100 px-3 py-1 font-bold text-fuchsia-900">{scenario.wow_tag}</span> : null}
          <span className="rounded-full bg-cyan-50 px-3 py-1 font-semibold text-cyan-900">Best fusion on test: {bestFusionRunId}</span>
        </div>
        <div className="mt-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-700">
          Scenario visuals are deterministic condition variants of the same base recording to match stress/missing-modality cases.
        </div>
        {bestFusionMetric ? (
          <div className="mt-2 rounded-lg border border-emerald-300 bg-emerald-50 px-3 py-2 text-sm font-bold text-emerald-900">
            TEST PR-AUC: {bestFusionMetric.run_id} = {bestFusionMetric.pr_auc.toFixed(3)}
            {visionUnimodalMetric ? ` vs UNI-V = ${visionUnimodalMetric.pr_auc.toFixed(3)}` : ""}
          </div>
        ) : null}
      </header>

      <section className="grid gap-3 lg:grid-cols-[2.2fr_1fr]">
        <div className="space-y-3">
          <div className={`grid gap-3 ${showVisionPanel ? "grid-cols-1 xl:grid-cols-2" : "grid-cols-1"}`}>
            {showVisionPanel ? (
              <VideoOverlayPanel
                label="Vision-only"
                videoRef={visionVideoRef}
                videoSrc={scenario.media.video}
                frame={frame}
                boxesFrom="vision"
                coordinateType={scenario.coordinate_type}
                sourceResolution={scenario.source_resolution}
                accent="#2563eb"
                performanceStep={performanceStep}
                badgeText={scenario.quality.vision === "degraded" ? "Vision signal (degraded)" : "Vision signal"}
                videoFilter={scenarioVideoFilter}
              />
            ) : null}
            <VideoOverlayPanel
              label="Fusion (F-GATE-NOQ score)"
              videoRef={fusionVideoRef}
              videoSrc={scenario.media.video}
              frame={frame}
              boxesFrom="fusion"
              coordinateType={scenario.coordinate_type}
              sourceResolution={scenario.source_resolution}
              accent="#0f766e"
              performanceStep={performanceStep}
              badgeText={fusionContextOnly ? "Fusion signal (audio+radar)" : "Fusion signal"}
              showBoxes={!fusionContextOnly}
              videoFilter={scenarioVideoFilter}
              contextNote={fusionContextOnly ? "Video shown for context only; vision modality is disabled." : undefined}
            />
          </div>

          {!visionAvailable ? (
            <div className="rounded-2xl border border-amber-300 bg-amber-50 px-3 py-2 text-sm font-semibold text-amber-900">
              Vision modality is missing in this scenario. Vision-only output is intentionally hidden.
            </div>
          ) : null}

          <div className="rounded-2xl border border-slate-300 bg-white p-3 shadow-sm">
            <div className="grid gap-2 md:grid-cols-[auto_1fr_auto_auto] md:items-center">
              <button
                onClick={onPlayPause}
                className="rounded-lg border border-slate-700 bg-slate-900 px-4 py-2 text-lg font-bold text-white"
              >
                {isPlaying ? "Pause" : "Play"}
              </button>

              <input
                type="range"
                min={0}
                max={duration}
                step={0.01}
                value={Math.min(duration, currentTime)}
                onChange={(e) => onScrub(Number(e.target.value))}
                className="w-full"
              />

              <label className="flex items-center gap-2 text-sm font-semibold text-slate-700">
                Perf
                <select
                  className="rounded border border-slate-300 bg-white px-2 py-1"
                  value={performanceStep}
                  onChange={(e) => setPerformanceStep(Number(e.target.value))}
                >
                  <option value={1}>N=1</option>
                  <option value={2}>N=2</option>
                  <option value={4}>N=4</option>
                </select>
              </label>

              <div className="text-right text-sm font-semibold text-slate-700">
                t={currentTime.toFixed(2)}s | frame {frameIdx}
              </div>
            </div>
          </div>

          {audio.length || radar.length ? (
            <div className={`grid gap-3 ${audio.length && radar.length ? "md:grid-cols-2" : "md:grid-cols-1"}`}>
              {audio.length ? <AudioWaveform samples={audio} currentTime={currentTime} /> : null}
              {radar.length ? <RadarSnapshot row={currentRadarRow} /> : null}
            </div>
          ) : null}
        </div>

        <aside className="space-y-3">
          <section className="rounded-2xl border border-slate-300 bg-white p-3 shadow-sm">
            <h3 className={`${posterMode ? "text-2xl" : "text-xl"} mb-2 font-black text-slate-900`}>Overall Leaderboard (Test)</h3>
            <div className="mb-2 rounded border border-slate-200 bg-slate-50 px-2 py-1 text-xs font-semibold text-slate-700">
              Includes unimodal and fusion methods, ranked by TEST PR-AUC.
            </div>
            <div className="space-y-2 text-sm">
              {leaderboardRows.map((m, idx) => {
                const isBest = idx === 0;
                return (
                  <div
                    key={`${m.run_id}-${m.split}`}
                    className={`rounded px-2 py-1 ${isBest ? "bg-emerald-50 font-bold text-emerald-900" : "bg-slate-50 text-slate-800"}`}
                  >
                    <div className="mb-1 flex items-center justify-between">
                      <span>{m.run_id}</span>
                      <span>PR-AUC {m.pr_auc.toFixed(3)}</span>
                    </div>
                    <div className="h-2 rounded bg-slate-200">
                      <div
                        className={`h-2 rounded ${isBest ? "bg-emerald-700" : "bg-slate-500"}`}
                        style={{ width: `${Math.max(0, Math.min(1, m.pr_auc / leaderboardMaxPrAuc)) * 100}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="rounded-2xl border border-slate-300 bg-white p-3 shadow-sm">
            <h3 className={`${posterMode ? "text-2xl" : "text-xl"} mb-2 font-black text-slate-900`}>Fusion Variants Comparison (Test only)</h3>
            <div className="mb-2 rounded border border-cyan-200 bg-cyan-50 px-2 py-1 text-xs font-semibold text-cyan-900">
              Compares fusion methods only: F-AVG, F-STATIC, F-GATE, F-GATE-NOQ.
            </div>
            {ablationFigure ? (
              <img src={ablationFigure} alt="Fusion variants comparison" className="mb-2 w-full rounded-lg border border-slate-200" />
            ) : null}
            {testFusionMetrics.length ? (
              <div className="space-y-1 text-sm">
                {testFusionMetrics.map((m) => {
                  const isBest = m.run_id === bestFusionRunId;
                  return (
                    <div key={m.run_id} className={`flex items-center justify-between rounded px-2 py-1 ${isBest ? "bg-cyan-50 font-bold text-cyan-900" : "bg-slate-50 text-slate-800"}`}>
                      <span>{m.run_id}</span>
                      <span>PR-AUC {m.pr_auc.toFixed(3)} | ROC-AUC {m.roc_auc.toFixed(3)}</span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-sm text-slate-500">Fusion metric table not available.</div>
            )}
          </section>

          <section className="rounded-2xl border border-slate-300 bg-white p-3 shadow-sm">
            <h3 className={`${posterMode ? "text-2xl" : "text-xl"} mb-2 font-black text-slate-900`}>Fusion Weights (Adaptive)</h3>
            <div className="space-y-2">
              {([
                { key: "vision", label: "Vision", q: scenario.quality.vision },
                { key: "audio", label: "Audio", q: scenario.quality.audio },
                { key: "radar", label: "Radar", q: scenario.quality.radar },
              ] as const).map((m) => {
                const weight = frame.weights[m.key];
                return (
                  <div key={m.key} className="space-y-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-semibold text-slate-800">{m.label}</span>
                      <StatusChip label={m.label} quality={m.q} />
                    </div>
                    <div className="h-3 overflow-hidden rounded-full bg-slate-200">
                      <div className="h-full bg-cyan-700" style={{ width: `${Math.max(0, Math.min(1, weight)) * 100}%` }} />
                    </div>
                    <div className="text-right text-xs font-semibold text-slate-700">{fmtPct(weight)}</div>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="rounded-2xl border border-slate-300 bg-white p-3 shadow-sm">
            <div className="mb-2 flex items-center justify-between">
              <h3 className={`${posterMode ? "text-2xl" : "text-xl"} font-black text-slate-900`}>Live Signal Snapshot</h3>
              {posterMode ? (
                <button
                  onClick={() => setShowComparison((v) => !v)}
                  className="rounded border border-slate-300 px-2 py-1 text-xs font-semibold"
                >
                  {showComparison ? "Collapse" : "Expand"}
                </button>
              ) : null}
            </div>
            <div className="mb-2 rounded border border-slate-200 bg-slate-50 px-2 py-1 text-xs font-semibold text-slate-700">
              Frame-level snapshot only. Use leaderboard/test tables for model ranking.
            </div>

            {showComparison ? (
              <div className="space-y-2 text-sm">
                {[
                  { name: "Vision", top: visionTop, prob: frame.modality_scores.vision, missing: scenario.quality.vision === "missing" },
                  { name: "Audio", top: audioTop, prob: frame.modality_scores.audio, missing: scenario.quality.audio === "missing" },
                  { name: "Radar", top: radarTop, prob: frame.modality_scores.radar, missing: scenario.quality.radar === "missing" },
                  { name: "Fusion", top: fusionTop, prob: frame.modality_scores.fusion, missing: false },
                ].map((row) => {
                  const signal = signalLabel(row.prob, row.missing);
                  return (
                    <div key={row.name} className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5">
                      <span className="font-semibold text-slate-800">{row.name}</span>
                      {row.missing ? (
                        <span className="text-slate-500">N/A</span>
                      ) : (
                        <span className="font-bold text-slate-900">
                          {row.top.label} {signal}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : null}
          </section>

          <section className="rounded-2xl border border-slate-300 bg-white p-3 shadow-sm">
            <h3 className={`${posterMode ? "text-2xl" : "text-xl"} mb-2 font-black text-slate-900`}>Why Fusion Helped</h3>
            <div className="mb-2 rounded-lg border border-cyan-200 bg-cyan-50 px-3 py-2 font-semibold text-cyan-900">{deltaMessage}</div>
            <div className="space-y-2">
              {fusionDistEntries.map(([label, value]) => (
                <div key={label} className="space-y-1">
                  <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
                    <span>{label}</span>
                    <span>{fmtPct(value)}</span>
                  </div>
                  <div className="h-2 rounded bg-slate-200">
                    <div className="h-2 rounded bg-teal-700" style={{ width: `${value * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </section>

      <section className="mt-3 rounded-2xl border border-slate-300 bg-white p-3 shadow-sm">
        <button
          className="w-full text-left text-lg font-black text-slate-900"
          onClick={() => setShowDownloadables((v) => !v)}
        >
          Downloadables {showDownloadables ? "(hide)" : "(show)"}
        </button>
        {showDownloadables ? (
          <div className="mt-2 flex flex-wrap gap-2 text-sm">
            {downloadables.length ? (
              downloadables.map((f) => (
                <a
                  key={f.path}
                  href={f.path}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded border border-slate-300 bg-slate-50 px-3 py-1.5 font-semibold text-slate-800 hover:border-cyan-700 hover:text-cyan-700"
                >
                  {f.name}
                </a>
              ))
            ) : (
              <span className="text-slate-500">No static figure exports detected.</span>
            )}
          </div>
        ) : null}
      </section>

      {showScript ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-2xl rounded-2xl border border-slate-300 bg-white p-5 shadow-2xl">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-2xl font-black text-slate-900">Demo script (60-120s)</h2>
              <button onClick={() => setShowScript(false)} className="rounded border border-slate-300 px-3 py-1 font-semibold text-slate-700">
                Close
              </button>
            </div>
            <ol className="space-y-2 text-base font-medium text-slate-800">
              {demo.demo_script.map((step, i) => (
                <li key={step} className="rounded-lg bg-slate-50 px-3 py-2">
                  <span className="mr-2 font-black text-cyan-800">{i + 1}.</span>
                  {step}
                </li>
              ))}
            </ol>
          </div>
        </div>
      ) : null}
    </main>
  );
}
