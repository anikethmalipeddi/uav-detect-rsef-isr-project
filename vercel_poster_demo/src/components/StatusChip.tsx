import { qualityColor } from "@/lib/classUtils";
import { Quality } from "@/lib/types";

type Props = {
  label: string;
  quality: Quality;
};

export default function StatusChip({ label, quality }: Props) {
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-semibold uppercase tracking-wide ${qualityColor(quality)}`}>
      {label}: {quality}
    </span>
  );
}
