import { AudioSample } from "@/lib/types";

type Props = {
  samples: AudioSample[];
  currentTime: number;
  quality?: "good" | "degraded" | "missing";
};

export default function AudioWaveform({ samples, currentTime, quality = "good" }: Props) {
  if (!samples.length) {
    return <div className="text-sm text-slate-500">Audio preview unavailable.</div>;
  }

  const maxT = samples[samples.length - 1]?.t_sec || 1;
  const cursorPct = Math.max(0, Math.min(100, (currentTime / maxT) * 100));
  const qualityText =
    quality === "missing" ? "Missing" : quality === "degraded" ? "Degraded" : "Good";

  const points = samples
    .map((s, idx) => {
      const x = (idx / Math.max(1, samples.length - 1)) * 100;
      const y = 50 - s.amplitude * 38;
      return `${x},${y}`;
    })
    .join(" ");

  const bars = Array.from({ length: 24 }, (_, i) => {
    const start = Math.floor((i / 24) * samples.length);
    const end = Math.max(start + 1, Math.floor(((i + 1) / 24) * samples.length));
    const slice = samples.slice(start, end);
    const energy = slice.reduce((acc, s) => acc + Math.abs(s.amplitude), 0) / Math.max(1, slice.length);
    return Math.max(0, Math.min(1, energy));
  });

  return (
    <div className="rounded-xl border border-slate-300 bg-white p-2">
      <div className="mb-1 flex items-center justify-between">
        <h4 className="text-sm font-semibold text-slate-800">Audio waveform</h4>
        <span className="rounded bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-700">
          {qualityText}
        </span>
      </div>
      <div className="relative h-20 w-full overflow-hidden rounded bg-slate-50">
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="h-full w-full">
          <line x1="0" y1="50" x2="100" y2="50" stroke="#cbd5e1" strokeWidth="0.8" />
          <polyline points={points} fill="none" stroke="#0f766e" strokeWidth="1.2" />
        </svg>
        <div className="absolute inset-y-0 w-0.5 bg-slate-900/70" style={{ left: `${cursorPct}%` }} />
      </div>
      <div className="mt-2">
        <div className="mb-1 text-xs font-semibold text-slate-600">Spectral activity</div>
        <div className="flex h-10 items-end gap-0.5 rounded bg-slate-50 p-1">
          {bars.map((b, i) => (
            <div key={i} className="flex-1 rounded-sm bg-cyan-700/80" style={{ height: `${Math.max(8, b * 100)}%` }} />
          ))}
        </div>
      </div>
    </div>
  );
}
