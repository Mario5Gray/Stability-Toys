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
    allowedHosts: ["enigma"],
    proxy: {
      "/generate": "http://localhost:4200",
      "/superres": "http://localhost:4200",
      "/v1": "http://localhost:4200",
      "/storage": "http://localhost:4200",
    },
  },
});
