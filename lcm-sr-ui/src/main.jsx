import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";
import { emitUiEvent } from "./utils/otelTelemetry";

if (!window.__appStartTime) {
  window.__appStartTime = performance.now();
}

if (!window.__otelErrorHooked) {
  window.__otelErrorHooked = true;
  window.addEventListener("error", (e) => {
    emitUiEvent("ui.error", {
      "ui.component": "global",
      "ui.action": "error",
      "error.message": e.message || "Unknown error",
      "error.filename": e.filename || "",
      "error.lineno": e.lineno || 0,
      "error.colno": e.colno || 0,
    });
  });
  window.addEventListener("unhandledrejection", (e) => {
    const reason = e.reason;
    emitUiEvent("ui.error", {
      "ui.component": "global",
      "ui.action": "unhandledrejection",
      "error.message": reason?.message || String(reason || "Unknown rejection"),
    });
  });
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
