import { RadarRow } from "@/lib/types";

type Props = {
  row: RadarRow | null;
};

export default function RadarSnapshot({ row }: Props) {
  if (!row) {
    return <div className="text-sm text-slate-500">Radar snapshot unavailable.</div>;
  }

  const bins = Object.keys(row)
    .filter((k) => k.startsWith("bin_"))
    .sort()
    .map((k) => Number(row[k]));

  const maxVal = Math.max(1e-6, ...bins);

  return (
    <div className="rounded-xl border border-slate-300 bg-white p-2">
      <h4 className="mb-1 text-sm font-semibold text-slate-800">Radar snapshot</h4>
      <div className="grid h-20 grid-cols-16 gap-0.5">
        {bins.map((v, idx) => {
          const alpha = Math.max(0.08, Math.min(1, v / maxVal));
          return (
            <div
              key={idx}
              className="rounded-sm"
              style={{
                background: `rgba(14, 116, 144, ${alpha.toFixed(3)})`,
              }}
            />
          );
        })}
      </div>
    </div>
  );
}
