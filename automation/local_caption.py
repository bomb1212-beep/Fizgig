"""Loopback-only llama.cpp vision client and owned batch-server lifecycle."""
import base64
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Local caption server redirected the request; refusing to send image elsewhere.")

class LocalClient:
    def __init__(self, cfg):
        self.cfg = cfg
        base = cfg["caption_base_url"].rstrip("/")
        parsed = urllib.parse.urlsplit(base)
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path != "/v1"):
            raise ValueError("caption_base_url must be a loopback HTTP URL ending in /v1.")
        self.base = base
        self.origin = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        self.host, self.port = parsed.hostname, parsed.port or 80
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def json_request(self, url, payload=None, timeout=2):
        req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=timeout) as response:
            return json.load(response)

    def ready(self):
        try:
            return self.json_request(self.origin + "/health").get("status") == "ok"
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def listening(self):
        try:
            with socket.create_connection((self.host, self.port), timeout=1):
                return True
        except OSError:
            return False

    def verify_model(self):
        models = self.json_request(self.base + "/models")
        ids = [item.get("id") for item in models.get("data", [])]
        if self.cfg["caption_model"] not in ids:
            raise RuntimeError(f"Local server has {ids}; expected {self.cfg['caption_model']}.")

    def caption(self, path, instructions):
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((1024, 1024))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
        image = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        payload = {"model": self.cfg["caption_model"], "stream": False,
                   "max_tokens": self.cfg["caption_max_output_tokens"], "temperature": self.cfg.get("caption_temperature", 0.2),
                   "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": instructions + "\nDescribe this photograph for LoRA training."},
                       {"type": "image_url", "image_url": {"url": image}}]}]}
        for attempt in range(3):
            try:
                data = self.json_request(self.base + "/chat/completions", payload,
                                         timeout=self.cfg.get("caption_request_timeout", 180))
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f"Local caption HTTP {exc.code}; inspect gemma_server.log.") from None
                time.sleep(2 ** (attempt + 1))
            except (urllib.error.URLError, TimeoutError) as exc:
                raise RuntimeError("Local Gemma connection failed or timed out; inspect gemma_server.log.") from exc
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("Local model returned no choices.")
        choice = choices[0]
        if choice.get("finish_reason") != "stop":
            raise RuntimeError("Caption did not finish; increase caption_max_output_tokens.")
        text = choice.get("message", {}).get("content")
        if not isinstance(text, str):
            raise RuntimeError("Local model returned no caption text.")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        if not text or "<think>" in text or "[THINK]" in text:
            raise RuntimeError("Only reasoning was returned; caption was not saved.")
        return self.cfg["trigger"] + ", " + " ".join(text.split()), data.get("usage", {})

class GemmaServer:
    """Only the process tree launched by this instance is stopped."""
    def __init__(self, cfg, run, cancelled=None):
        self.cfg, self.run = cfg, Path(run)
        self.client = LocalClient(cfg)
        self.process = self.log = None
        self.external = False
        self.cancelled = cancelled or (lambda: False)

    def __enter__(self):
        if self.client.ready():
            self.client.verify_model()
            self.external = True
            print("Using already-running Gemma server; it will remain running.", flush=True)
            return self.client
        if self.client.listening():
            raise RuntimeError("Caption port is occupied by an unready server; wait or close that server.")
        if not self.cfg.get("caption_auto_start", True):
            raise RuntimeError("Gemma server is not running; start caption_start_script first.")
        mode = self.cfg.get("caption_launch_mode", "batch")
        if mode not in ("batch", "direct"):
            raise ValueError("caption_launch_mode must be batch or direct.")
        executable_key = "caption_start_script" if mode == "batch" else "caption_server_exe"
        for key in (executable_key, "caption_model_file", "caption_mmproj_file"):
            if not Path(self.cfg[key]).is_file():
                raise ValueError(f"Missing {key}: {self.cfg[key]}")
        if os.name != "nt":
            raise RuntimeError("The configured Gemma batch launcher requires Windows.")
        self.run.mkdir(parents=True, exist_ok=True)
        self.log = (self.run / "gemma_server.log").open("a", encoding="utf-8")
        if mode == "batch":
            script = self.cfg["caption_start_script"]
            command = ["cmd.exe", "/d", "/c", script]
            working_dir = str(Path(script).parent)
        else:
            context = int(self.cfg.get("caption_context_size", 65536))
            if context <= 0:
                raise ValueError("caption_context_size must be positive.")
            command = [self.cfg["caption_server_exe"], "-m", self.cfg["caption_model_file"],
                "--mmproj", self.cfg["caption_mmproj_file"], "-c", str(context),
                "--parallel", "1", "-ngl", "99", "--flash-attn", "on", "-ctk", "q8_0", "-ctv", "q8_0",
                "--kv-unified", "--jinja", "--reasoning", "off", "--host", self.client.host,
                "--port", str(self.client.port), "--alias", self.cfg["caption_model"]]
            working_dir = str(Path(self.cfg["caption_server_exe"]).parent)
        try:
            self.process = subprocess.Popen(command, cwd=working_dir,
                stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW)
            print(f"Starting local Gemma (owned launcher PID {self.process.pid})...", flush=True)
            deadline = time.monotonic() + self.cfg.get("caption_startup_timeout", 300)
            last_update = 0
            while time.monotonic() < deadline:
                if self.cancelled():
                    raise RuntimeError("Model loading paused by user.")
                if self.process.poll() is not None:
                    raise RuntimeError(f"Gemma launcher exited; inspect {self.run / 'gemma_server.log'}")
                if self.client.ready():
                    self.client.verify_model()
                    print("Gemma ready for image captions.", flush=True)
                    return self.client
                if time.monotonic() - last_update > 30:
                    print("Waiting for Gemma model load; progress in gemma_server.log...", flush=True)
                    last_update = time.monotonic()
                time.sleep(1)
            raise RuntimeError(f"Gemma startup timed out; inspect {self.run / 'gemma_server.log'}")
        except BaseException:
            self.close()
            raise

    def close(self):
        try:
            if self.process is not None and self.process.poll() is None:
                # PID is from this instance's Popen handle, never from a process-name search.
                result = subprocess.run(["taskkill.exe", "/PID", str(self.process.pid), "/T", "/F"],
                    capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                if result.returncode != 0 and self.process.poll() is None:
                    raise RuntimeError("Could not stop owned Gemma server; training will not start.")
                self.process.wait(timeout=15)
                deadline = time.monotonic() + 15
                while self.client.listening() and time.monotonic() < deadline:
                    time.sleep(0.5)
                if self.client.listening():
                    raise RuntimeError("Gemma port remains occupied; training will not start.")
                print("Owned Gemma server stopped; GPU released for Qwen training.", flush=True)
        finally:
            if self.log is not None:
                self.log.close()
                self.log = None

    def __exit__(self, exc_type, exc, tb):
        self.close()
