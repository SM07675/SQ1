import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: process.env.SATQUERY_DEV_API_TARGET || "http://localhost:8000", changeOrigin: true },
      "/health": { target: process.env.SATQUERY_DEV_API_TARGET || "http://localhost:8000", changeOrigin: true },
      "/artifacts": { target: process.env.SATQUERY_DEV_API_TARGET || "http://localhost:8000", changeOrigin: true }
    }
  }
});

