// vite.config.js
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
    allowedHosts: process.env.VITE_ALLOWED_HOSTS 
      ? process.env.VITE_ALLOWED_HOSTS.split(',')
      : ["mindgate", "enigma", "node2", "enigma:5173"],
    host: true,
    watch: {
      usePolling: true,
      interval: 300,
    },    
    proxy: {
      "/generate": "http://localhost:4200",
      "/superres": "http://localhost:4200",
      "/v1": {
        target: "http://localhost:4200",
        ws: true,
      },
      "/storage": "http://localhost:4200",
      "/dreams": "http://localhost:4200",
      "/api": "http://localhost:4200",
    },
  },
});
