import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run dev` serves the UI on :5173 and forwards /api to `tennis serve` on :8731.
// `npm run build` writes into the Python package, which serves it in production.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../tennis/api/static", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:8731" } },
});
