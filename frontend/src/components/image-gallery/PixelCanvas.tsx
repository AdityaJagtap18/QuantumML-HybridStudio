import { useEffect, useRef } from "react";

export function PixelCanvas({
  pixels,
  pixelMax,
  size = 64,
  highlight = false,
}: {
  pixels: number[][];
  pixelMax: number;
  size?: number;
  highlight?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rows = pixels.length;
  const cols = pixels[0]?.length ?? 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const cellW = size / cols;
    const cellH = size / rows;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const v = Math.min(1, Math.max(0, pixels[i][j] / pixelMax));
        // Dark ink on light background: 0 -> white-ish canvas color, max -> ink.
        const channel = Math.round(255 - v * (255 - 32));
        ctx.fillStyle = `rgb(${channel},${channel},${channel})`;
        ctx.fillRect(j * cellW, i * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }, [pixels, pixelMax, size, rows, cols]);

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      className={`${highlight ? "ring-2 ring-accent" : "border border-border"}`}
      style={{ width: size, height: size, imageRendering: "pixelated" }}
    />
  );
}
