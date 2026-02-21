export type Quality = "good" | "degraded" | "missing";

export type ClassLabel = "DRONE" | "AIRPLANE" | "BIRD" | "HELICOPTER";

export type Box = {
  x: number;
  y: number;
  w: number;
  h: number;
  class: ClassLabel;
  conf: number;
};

export type FrameRow = {
  frame_idx: number;
  t: number;
  vision: {
    boxes: Box[];
    score: number;
  };
  fusion: {
    boxes: Box[];
    score: number;
  };
  modality_scores: {
    vision: number;
    audio: number;
    radar: number;
    fusion: number;
  };
  weights: {
    vision: number;
    audio: number;
    radar: number;
  };
};

export type Scenario = {
  id: string;
  title: string;
  demonstrates: string;
  wow: boolean;
  wow_tag: string;
  ground_truth: {
    label: string;
    class: ClassLabel;
  };
  media: {
    video: string;
    audio?: string;
    frames_prefix?: string;
  };
  quality: {
    vision: Quality;
    audio: Quality;
    radar: Quality;
  };
  coordinate_type: "pixel" | "normalized";
  source_resolution: {
    width: number;
    height: number;
  };
  focus_frame: number;
  summaries: {
    vision: { class_probs: Record<ClassLabel, number> };
    audio: { class_probs: Record<ClassLabel, number> };
    radar: { class_probs: Record<ClassLabel, number> };
    fusion: { class_probs: Record<ClassLabel, number>; model: string };
  };
  frames: FrameRow[];
};

export type DemoData = {
  meta: {
    name: string;
    version: string;
    fusion_primary_model: string;
    fusion_weights_source: string;
    notes: string;
    default_scenarios: string[];
    wow_scenarios: string[];
  };
  demo_script: string[];
  scenarios: Scenario[];
};

export type Downloadable = {
  name: string;
  path: string;
};

export type AudioSample = {
  t_sec: number;
  amplitude: number;
};

export type RadarRow = {
  frame_idx: number;
  [key: string]: number;
};
