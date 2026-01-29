# invokers/comfy_client.py
from __future__ import annotations

import json
import time
import uuid
import requests
import websocket  # websocket-client
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

class ComfyUIError(RuntimeError):
    pass

@dataclass(frozen=True)
class ComfyFileRef:
    filename: str
    subfolder: str = ""
    type: str = "output"  # output|input|temp

    def view_params(self) -> Dict[str, str]:
        p = {"filename": self.filename, "type": self.type}
        if self.subfolder:
            p["subfolder"] = self.subfolder
        return p


@dataclass(frozen=True)
class ComfyInvokeResult:
    prompt_id: str
    history: Json
    outputs: List[ComfyFileRef]


# invokers/comfy_client.py
@dataclass
class OutputRef:
    filename: str
    subfolder: str
    type: str  # "output" typically

import requests
from typing import Optional, Dict

class ComfyUIInvoker:
    def __init__(
        self,
        base_url: str,
        verify_tls: bool = True,
        timeout_s: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.verify_tls = verify_tls

        # Used by _get/_post
        self.timeout_s = float(timeout_s)
        self.headers = dict(headers or {})

        # Restore session (connection pooling)
        self.session = session or requests.Session()
    # --- existing: upload_image(...) etc ---

    def submit_prompt(self, prompt: Dict[str, Any], client_id: str) -> str:
        """
        POST /prompt => returns prompt_id
        """
        url = f"{self.base_url}/prompt"
        r = requests.post(url, json={"prompt": prompt, "client_id": client_id}, verify=self.verify_tls, timeout=30)
        r.raise_for_status()
        data = r.json()
        # ComfyUI returns {"prompt_id": "...", ...}
        return data["prompt_id"]

    def open_ws(self, client_id: str):
        """
        WS URL mirrors base_url scheme/host.
        """
        # base_url like https://node2:8189
        if self.base_url.startswith("https://"):
            ws_url = "wss://" + self.base_url[len("https://"):]
        elif self.base_url.startswith("http://"):
            ws_url = "ws://" + self.base_url[len("http://"):]
        else:
            ws_url = "ws://" + self.base_url

        ws_url = f"{ws_url}/ws?clientId={client_id}"
        return websocket.create_connection(ws_url, timeout=30)

    def wait_with_node_progress(
        self,
        ws,
        prompt_id: str,
        on_node: callable,
        max_wait_s: float = 900,
    ) -> None:
        """
        Listen for 'executing' events until node == None for matching prompt_id.
        Calls on_node(node_id_str_or_none) as events arrive.
        """
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            raw = ws.recv()
            # WS may send binary preview frames; ignore those
            if isinstance(raw, (bytes, bytearray)):
                continue
            msg = json.loads(raw)
            if msg.get("type") != "executing":
                continue

            data = msg.get("data") or {}
            if data.get("prompt_id") != prompt_id:
                continue

            node = data.get("node")  # string id or None
            on_node(node)

            if node is None:
                return

        raise TimeoutError(f"Timed out waiting for prompt_id={prompt_id}")

    def get_history_outputs(self, prompt_id: str) -> List[OutputRef]:
        """
        GET /history/{prompt_id} and extract output image refs.
        """
        url = f"{self.base_url}/history/{prompt_id}"
        r = requests.get(url, verify=self.verify_tls, timeout=30)
        r.raise_for_status()
        hist = r.json()

        # Shape: {prompt_id: {outputs: {node_id: {images:[...]}}}}
        item = hist.get(prompt_id) or {}
        outputs = item.get("outputs") or {}
        refs: List[OutputRef] = []

        for _node_id, out in outputs.items():
            for img in (out.get("images") or []):
                refs.append(
                    OutputRef(
                        filename=img.get("filename", ""),
                        subfolder=img.get("subfolder", ""),
                        type=img.get("type", "output"),
                    )
                )
        return refs
    # --- public API ---

    def upload_image(self, content: bytes, filename: str, image_type: str = "input") -> Json:
        """
        ComfyUI: POST /upload/image (multipart)
        """
        files = {"image": (filename, content, "application/octet-stream")}
        data = {"type": image_type}
        r = self._post("/upload/image", data_body=data, files=files)
        return self._json_or_raise(r, "upload_image")

    def invoke(
        self,
        prompt_graph: Json,
        client_id: Optional[str] = None,
        poll_interval_s: float = 0.5,
        max_wait_s: float = 900.0,
    ) -> ComfyInvokeResult:
        prompt_id = self.queue_prompt(prompt_graph, client_id=client_id)
        history = self.wait_for_history(prompt_id, poll_interval_s=poll_interval_s, max_wait_s=max_wait_s)
        outputs = self.extract_outputs(history, prompt_id)
        return ComfyInvokeResult(prompt_id=prompt_id, history=history, outputs=outputs)

    def queue_prompt(self, prompt_graph: Json, client_id: Optional[str] = None) -> str:
        payload = {"prompt": prompt_graph, "client_id": client_id or f"invokers-{uuid.uuid4()}"}
        r = self._post("/prompt", json_body=payload)
        data = self._json_or_raise(r, "queue_prompt")
        pid = data.get("prompt_id")
        if not pid:
            raise ComfyUIError(f"/prompt missing prompt_id: {data}")
        return str(pid)

    def get_history(self, prompt_id: str) -> Json:
        r = self._get(f"/history/{prompt_id}")
        return self._json_or_raise(r, "get_history")

    def wait_for_history(self, prompt_id: str, poll_interval_s: float, max_wait_s: float) -> Json:
        deadline = time.time() + max_wait_s
        last = None
        while time.time() < deadline:
            hist = self.get_history(prompt_id)
            last = hist
            if isinstance(hist, dict) and prompt_id in hist:
                node_graph = hist[prompt_id]
                self._raise_if_history_error(node_graph)
                if self._history_has_outputs(node_graph):
                    return hist
            time.sleep(poll_interval_s)
        raise ComfyUIError(f"Timed out waiting for {prompt_id}. Last={last}")

    def extract_outputs(self, history: Json, prompt_id: str) -> List[ComfyFileRef]:
        if prompt_id not in history:
            return []
        node_graph = history[prompt_id]
        self._raise_if_history_error(node_graph)

        out_map = node_graph.get("outputs", {})
        if not isinstance(out_map, dict):
            return []

        refs: List[ComfyFileRef] = []
        for node_out in out_map.values():
            if not isinstance(node_out, dict):
                continue
            for key in ("images", "gifs", "audio", "files"):
                vals = node_out.get(key)
                if not isinstance(vals, list):
                    continue
                for item in vals:
                    if not isinstance(item, dict):
                        continue
                    fn = item.get("filename")
                    if not fn:
                        continue
                    refs.append(
                        ComfyFileRef(
                            filename=str(fn),
                            subfolder=str(item.get("subfolder") or ""),
                            type=str(item.get("type") or "output"),
                        )
                    )

        uniq: Dict[Tuple[str, str, str], ComfyFileRef] = {}
        for r in refs:
            uniq[(r.filename, r.subfolder, r.type)] = r
        return list(uniq.values())

    # --- HTTP helpers ---

    def _get(self, path: str, params: Optional[Dict[str, str]] = None, stream: bool = False) -> requests.Response:
        return self.session.get(
            self.base_url + path,
            params=params,
            timeout=self.timeout_s,
            headers=self.headers,
            verify=self.verify_tls,
            stream=stream,
        )

    def _post(
        self,
        path: str,
        json_body: Optional[Json] = None,
        data_body: Optional[Dict[str, str]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        return self.session.post(
            self.base_url + path,
            json=json_body,
            data=data_body,
            files=files,
            timeout=self.timeout_s,
            headers=self.headers,
            verify=self.verify_tls,
        )

    def _json_or_raise(self, r: requests.Response, where: str) -> Json:
        if not (200 <= r.status_code < 300):
            raise ComfyUIError(f"{where} HTTP {r.status_code}: {r.text[:1000]}")
        try:
            return r.json()
        except Exception as e:
            raise ComfyUIError(f"{where} JSON decode error: {e}; body={r.text[:1000]}")

    def _history_has_outputs(self, node_graph: Any) -> bool:
        if not isinstance(node_graph, dict):
            return False
        outs = node_graph.get("outputs")
        if not isinstance(outs, dict) or not outs:
            return False
        for v in outs.values():
            if not isinstance(v, dict):
                continue
            for k in ("images", "gifs", "audio", "files"):
                if isinstance(v.get(k), list) and len(v.get(k)) > 0:
                    return True
        return False

    def _raise_if_history_error(self, node_graph: Any) -> None:
        if not isinstance(node_graph, dict):
            return
        status = node_graph.get("status")
        if isinstance(status, dict):
            if status.get("status_str") == "error" or status.get("error") or status.get("exception_message"):
                raise ComfyUIError(f"ComfyUI error: {status}")
        if node_graph.get("error"):
            raise ComfyUIError(f"ComfyUI error: {node_graph.get('error')}")