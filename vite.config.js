import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  base: "./",
  root: path.resolve(__dirname, "frontend"),
  plugins: [react()],
  build: {
    outDir: path.resolve(__dirname, "frontend", "dist"),
    emptyOutDir: true,
  },
});
