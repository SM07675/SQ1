import React, { useEffect, useRef } from "react";
import { GradientWave } from "@/components/ui/gradient-wave";

/**
 * LiquidAuroraBackground
 * Creates a full-viewport animated background combining:
 * - GradientWave dynamic canvas with multi-harmonic flowing liquid waves
 * - 6 large blurred ambient radial nodes
 * - 3 translucent SVG wave crests
 * - Atmospheric floating particles
 *
 * All layers are pointer-events-none, fixed, and never affect page layout.
 * Respects prefers-reduced-motion.
 */

export const LiquidAuroraBackground: React.FC<{ theme?: "light" | "dark" }> = ({
  theme = "light",
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);

  // Subtle ambient particles
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReduced) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    const resize = () => {
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      ctx.scale(dpr, dpr);
    };
    resize();
    window.addEventListener("resize", resize);

    interface Particle {
      x: number;
      y: number;
      vx: number;
      vy: number;
      size: number;
      opacity: number;
    }

    const particleCount = theme === "dark" ? 35 : 18;
    const maxOpacity = theme === "dark" ? 0.2 : 0.06;
    const particles: Particle[] = Array.from({ length: particleCount }, () => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.15,
      vy: (Math.random() - 0.5) * 0.12,
      size: Math.random() * 2 + 0.5,
      opacity: Math.random() * maxOpacity + 0.01,
    }));

    let lastTime = 0;
    const targetFPS = 25;
    const interval = 1000 / targetFPS;

    const animate = (time: number) => {
      if (document.visibilityState === "hidden") {
        rafRef.current = requestAnimationFrame(animate);
        return;
      }

      if (time - lastTime < interval) {
        rafRef.current = requestAnimationFrame(animate);
        return;
      }
      lastTime = time;

      const w = window.innerWidth;
      const h = window.innerHeight;
      ctx.clearRect(0, 0, w, h);

      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = w;
        if (p.x > w) p.x = 0;
        if (p.y < 0) p.y = h;
        if (p.y > h) p.y = 0;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle =
          theme === "dark"
            ? `rgba(200, 210, 255, ${p.opacity})`
            : `rgba(140, 160, 220, ${p.opacity})`;
        ctx.fill();
      });

      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [theme]);

  return (
    <div className="liquid-aurora-bg" aria-hidden="true">
      {/* 1. GradientWave WebGL Component */}
      <GradientWave
        theme={theme}
        colors={
          theme === "dark"
            ? ["#030712", "#0f172a", "#1e1b4b", "#0f766e", "#1e293b", "#0284c7"]
            : ["#38bdf8", "#ffffff", "#bae6fd", "#ede9fe", "#38bdf8", "#f0fdf4"]
        }
        isPlaying={true}
        shadowPower={theme === "dark" ? 8 : 6}
        darkenTop={false}
        noiseSpeed={0.000015}
        noiseFrequency={[0.00015, 0.0008]}
        deform={{ incline: 0.45, noiseAmp: 280, noiseFlow: 4 }}
      />

      {/* 2. Atmospheric subtle floating particles */}
      <canvas ref={canvasRef} className="aurora-particles-canvas" />
    </div>
  );
};

export default LiquidAuroraBackground;
