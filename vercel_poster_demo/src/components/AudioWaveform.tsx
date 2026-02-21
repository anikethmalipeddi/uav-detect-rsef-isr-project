import { AudioSample } from "@/lib/types";

type Props = {
  samples: AudioSample[];
  currentTime: number;
};

export default function AudioWaveform({ samples, currentTime }: Props) {
  if (!samples.length) {
    return <div className="text-sm text-slate-500">Audio preview unavailable.</div>;
  }

  const maxT = samples[samples.length - 1]?.t_sec || 1;
  const cursorPct = Math.max(0, Math.min(100, (currentTime / maxT) * 100));

  const points = samples
    .map((s, idx) => {
      const x = (idx / Math.max(1, samples.length - 1)) * 100;
      const y = 50 - s.amplitude * 38;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="rounded-xl border border-slate-300 bg-white p-2">
      <h4 className="mb-1 text-sm font-semibold text-slate-800">Audio waveform</h4>
      <div className="relative h-20 w-full overflow-hidden rounded bg-slate-50">
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="h-full w-full">
          <line x1="0" y1="50" x2="100" y2="50" stroke="#cbd5e1" strokeWidth="0.8" />
          <polyline points={points} fill="none" stroke="#0f766e" strokeWidth="1.2" />
        </svg>
        <div className="absolute inset-y-0 w-0.5 bg-slate-900/70" style={{ left: `${cursorPct}%` }} />
      </div>
    </div>
  );
}
