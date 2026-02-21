import { ClassLabel } from "@/lib/types";

const NON_DRONE_PRIORS: Record<Exclude<ClassLabel, "DRONE">, number> = {
  AIRPLANE: 0.3247,
  BIRD: 0.3917,
  HELICOPTER: 0.2836,
};

export function toClassDistribution(droneProb: number): Record<ClassLabel, number> {
  const pDrone = Math.max(0, Math.min(1, droneProb));
  const pNonDrone = 1 - pDrone;
  return {
    DRONE: pDrone,
    AIRPLANE: pNonDrone * NON_DRONE_PRIORS.AIRPLANE,
    BIRD: pNonDrone * NON_DRONE_PRIORS.BIRD,
    HELICOPTER: pNonDrone * NON_DRONE_PRIORS.HELICOPTER,
  };
}

export function topClassFromDroneProb(droneProb: number): { label: ClassLabel; conf: number } {
  const dist = toClassDistribution(droneProb);
  const entries = Object.entries(dist) as [ClassLabel, number][];
  entries.sort((a, b) => b[1] - a[1]);
  return { label: entries[0][0], conf: entries[0][1] };
}

export function classEntries(desc: Record<ClassLabel, number>): [ClassLabel, number][] {
  const entries = Object.entries(desc) as [ClassLabel, number][];
  entries.sort((a, b) => b[1] - a[1]);
  return entries;
}

export function qualityColor(q: "good" | "degraded" | "missing"): string {
  if (q === "good") return "bg-emerald-100 text-emerald-900 border-emerald-300";
  if (q === "degraded") return "bg-amber-100 text-amber-900 border-amber-300";
  return "bg-rose-100 text-rose-900 border-rose-300";
}
