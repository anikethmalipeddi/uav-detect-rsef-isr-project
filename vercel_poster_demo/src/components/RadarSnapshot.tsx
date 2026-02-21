import { RadarRow } from "@/lib/types";

type Props = {
  rows: RadarRow[];
  currentFrameIdx: number;
  quality?: "good" | "degraded" | "missing";
};

export default function RadarSnapshot({ rows, currentFrameIdx, quality = "good" }: Props) {
  if (!rows.length) {
    return <div className="text-sm text-slate-500">Radar snapshot unavailable.</div>;
  }

  const bins = Object.keys(rows[0])
    .filter((k) => k.startsWith("bin_"))
    .sort()
    .map((k) => k);

  const centerIdx = Math.max(
    0,
    rows.findIndex((r) => Number(r.frame_idx) >= currentFrameIdx),
  );
  const windowSize = 24;
  const start = Math.max(0, centerIdx - windowSize + 1);
  const slice = rows.slice(start, start + windowSize);
  const qualityText =
    quality === "missing" ? "Missing" : quality === "degraded" ? "Degraded" : "Good";

  let maxVal = 1e-6;
  for (const r of slice) {
    for (const b of bins) {
      maxVal = Math.max(maxVal, Number(r[b]));
    }
  }

  return (
    <div className="rounded-xl border border-slate-300 bg-white p-2">
      <div className="mb-1 flex items-center justify-between">
        <h4 className="text-sm font-semibold text-slate-800">Radar activity heatmap</h4>
        <span className="rounded bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-700">
          {qualityText}
        </span>
      </div>
      <div
        className="grid h-24 gap-0.5 rounded bg-slate-50 p-1"
        style={{ gridTemplateColumns: `repeat(${bins.length}, minmax(0,1fr))` }}
      >
        {slice.map((r, ridx) =>
          bins.map((b) => {
            const v = Number(r[b]);
            const alpha = Math.max(0.08, Math.min(1, v / maxVal));
            return (
              <div
                key={`${ridx}-${b}`}
                className="rounded-sm"
                style={{
                  background: `rgba(14, 116, 144, ${alpha.toFixed(3)})`,
                }}
              />
            );
          }),
        )}
      </div>
      <div className="mt-1 text-xs font-semibold text-slate-600">Recent frames up to current time</div>
    </div>
  );
}
