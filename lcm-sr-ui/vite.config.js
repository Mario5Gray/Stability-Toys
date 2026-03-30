// vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

const apiTarget = process.env.VITE_API_TARGET ?? "http://localhost:4200";


export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    allowedHosts: process.env.VITE_ALLOWED_HOSTS 
      ? process.env.VITE_ALLOWED_HOSTS.split(',')
      : ["mindgate", "enigma", "node2", "enigma:5173","mindgate:4200"],
    host: true,
    watch: {
      usePolling: true,
      interval: 300,
    },    
    proxy: {
      "/generate": apiTarget,
      "/superres": apiTarget,
      "/v1": {
        target: apiTarget,
        ws: true,
      },
      "/storage": apiTarget,
      "/dreams": apiTarget,
      "/api": apiTarget,
    },
  },
});
