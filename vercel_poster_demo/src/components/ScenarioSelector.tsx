import { Scenario } from "@/lib/types";

type Props = {
  scenarios: Scenario[];
  selectedId: string;
  onSelect: (id: string) => void;
  posterMode: boolean;
};

export default function ScenarioSelector({ scenarios, selectedId, onSelect, posterMode }: Props) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-2">
      {scenarios.map((scenario) => {
        const active = selectedId === scenario.id;
        return (
          <button
            key={scenario.id}
            onClick={() => onSelect(scenario.id)}
            className={`rounded-full border px-3 py-1.5 font-semibold transition ${posterMode ? "text-base" : "text-sm"} ${
              active
                ? "border-cyan-700 bg-cyan-700 text-white"
                : "border-slate-400 bg-white text-slate-800 hover:border-cyan-600 hover:text-cyan-700"
            }`}
            title={scenario.demonstrates}
          >
            {scenario.title}
            {scenario.wow ? ` (${scenario.wow_tag})` : ""}
          </button>
        );
      })}
    </div>
  );
}
