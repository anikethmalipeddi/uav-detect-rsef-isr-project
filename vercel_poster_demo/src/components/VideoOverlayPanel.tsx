"use client";

import { RefObject, useEffect, useRef } from "react";
import { Box, FrameRow } from "@/lib/types";

type Props = {
  label: string;
  videoRef: RefObject<HTMLVideoElement>;
  videoSrc: string;
  frame: FrameRow | null;
  boxesFrom: "vision" | "fusion";
  coordinateType: "pixel" | "normalized";
  sourceResolution: { width: number; height: number };
  accent: string;
  performanceStep: number;
  hidden?: boolean;
};

function toCanvasBox(
  b: Box,
  canvasW: number,
  canvasH: number,
  coordType: "pixel" | "normalized",
  srcW: number,
  srcH: number,
) {
  if (coordType === "normalized") {
    return {
      x: b.x * canvasW,
      y: b.y * canvasH,
      w: b.w * canvasW,
      h: b.h * canvasH,
    };
  }
  return {
    x: (b.x / srcW) * canvasW,
    y: (b.y / srcH) * canvasH,
    w: (b.w / srcW) * canvasW,
    h: (b.h / srcH) * canvasH,
  };
}

export default function VideoOverlayPanel({
  label,
  videoRef,
  videoSrc,
  frame,
  boxesFrom,
  coordinateType,
  sourceResolution,
  accent,
  performanceStep,
  hidden = false,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const lastRenderedRef = useRef<number>(-1);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const wrapper = wrapperRef.current;
    if (!video || !canvas || !wrapper) return;

    const resize = () => {
      const rect = video.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        canvas.width = Math.round(rect.width);
        canvas.height = Math.round(rect.height);
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;
      }
    };

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrapper);
    window.addEventListener("resize", resize);

    return () => {
      ro.disconnect();
      window.removeEventListener("resize", resize);
    };
  }, [videoRef]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame) return;
    if (performanceStep > 1 && frame.frame_idx % performanceStep !== 0 && lastRenderedRef.current >= 0) {
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const boxes = boxesFrom === "vision" ? frame.vision.boxes : frame.fusion.boxes;
    for (const box of boxes) {
      const c = toCanvasBox(
        box,
        canvas.width,
        canvas.height,
        coordinateType,
        sourceResolution.width,
        sourceResolution.height,
      );
      ctx.strokeStyle = accent;
      ctx.lineWidth = 2;
      ctx.strokeRect(c.x, c.y, c.w, c.h);
      ctx.fillStyle = accent;
      const text = `${box.class} ${(box.conf * 100).toFixed(1)}%`;
      ctx.font = "bold 12px sans-serif";
      const tw = ctx.measureText(text).width + 8;
      const th = 18;
      ctx.fillRect(c.x, Math.max(0, c.y - th), tw, th);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, c.x + 4, Math.max(12, c.y - 5));
    }

    lastRenderedRef.current = frame.frame_idx;
  }, [frame, boxesFrom, accent, coordinateType, sourceResolution.height, sourceResolution.width, performanceStep]);

  const topScore = frame
    ? boxesFrom === "vision"
      ? frame.modality_scores.vision
      : frame.modality_scores.fusion
    : 0;

  return (
    <section className={`${hidden ? "hidden" : "flex"} min-w-0 flex-1 flex-col rounded-2xl border border-slate-300 bg-white p-3 shadow-sm`}>
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-lg font-bold text-slate-900">{label}</h3>
        <div className="rounded-lg bg-slate-100 px-3 py-1 text-lg font-black text-slate-900">
          DRONE {(topScore * 100).toFixed(1)}%
        </div>
      </div>

      <div ref={wrapperRef} className="relative w-full overflow-hidden rounded-xl border border-slate-300 bg-black">
        <video
          ref={videoRef}
          src={videoSrc}
          preload="auto"
          muted
          playsInline
          className="block h-auto w-full"
        />
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-0" />
      </div>
    </section>
  );
}
