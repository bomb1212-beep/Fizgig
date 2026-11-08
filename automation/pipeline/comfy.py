"""A ComfyUI client for the pipeline, with a journal so a crash cannot silently generate twice.

The tool2 client (`qwen 2.1 edit/dataset_batch.py`) is the model for the request shapes here; this one adds the parts
the pipeline needs and tool2 was written without:

* **The prompt_id is on disk before the caller sees it.** `submit()` writes the id to the journal and only then
  returns, so a client that dies straight after a submission finds the id again instead of submitting a second time.
* **Images are written to a temporary name and renamed**, so a killed process leaves no half PNG that a later run
  would skip as "already there".
* **Filenames are resolved through `/object_info`.** A configured absolute path becomes the name ComfyUI itself lists
  for that model; a path ComfyUI cannot see fails here, with the folder it should be in, rather than at sampling time.

Nothing here starts or stops the server. `/free` only asks ComfyUI to release VRAM; it never kills a process.
"""
from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

# Which /object_info entry lists the files of each kind, and the widget that holds them.
FILENAME_LISTS = {
    "unet": ("UNETLoader", "unet_name"),
    "lora": ("LoraLoaderModelOnly", "lora_name"),
    "vae": ("VAELoader", "vae_name"),
    "clip": ("CLIPLoader", "clip_name"),
}


class ComfyError(RuntimeError):
    """ComfyUI answered, but not with a usable result."""


class Cancelled(RuntimeError):
    """The caller asked to stop while a prompt was queued or running."""


def _atomic_write(path, data):
    """Write `data` to `path` through a temporary file in the same folder. Returns the path as a str."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex[:8])
    with open(temp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return str(path)


class Journal:
    """One small JSON file per submitted prompt, written atomically.

    A file per prompt keeps the record independent of append ordering: a crash can only lose the prompt whose file
    was being written, and that prompt was never submitted (the record is written before the POST). The submit itself
    is preceded by an *intent* record keyed by client id - the prompt id does not exist until ComfyUI answers, so the
    intent is what makes a lost response recoverable instead of a silent double submission (spec R6).
    """

    SENDING, SUBMITTED, DONE, FAILED, CANCELLED = "sending", "submitted", "done", "failed", "cancelled"

    def __init__(self, folder):
        self.folder = Path(folder)

    def path(self, prompt_id):
        return self.folder / (prompt_id + ".json")

    def intent_path(self, client_id):
        return self.folder / f"intent-{client_id}.json"

    def record(self, prompt_id, **meta):
        data = {"prompt_id": prompt_id, "status": self.SUBMITTED, "at": time.time(), **meta}
        _atomic_write(self.path(prompt_id), json.dumps(data, ensure_ascii=False, indent=1).encode())
        return data

    def record_intent(self, client_id, **meta):
        """The submit's intent, on disk before the POST: enough to know a submission may be in flight."""
        data = {"client_id": client_id, "status": self.SENDING, "at": time.time(), **meta}
        _atomic_write(self.intent_path(client_id), json.dumps(data, ensure_ascii=False, indent=1).encode())
        return data

    def commit_intent(self, client_id, prompt_id, **meta):
        """The POST answered: the intent becomes the prompt's own record and the intent file goes away."""
        intent = self.read_intent(client_id) or {}
        carried = {k: v for k, v in intent.items() if k not in ("status", "at", "updated_at")}
        self.record(prompt_id, **{**carried, **meta, "client_id": client_id})
        self.intent_path(client_id).unlink(missing_ok=True)
        return self.read(prompt_id)

    def drop_intent(self, client_id, status=FAILED, **extra):
        """A submit that never happened (the server refused the graph): the intent is closed, not left dangling."""
        intent = self.read_intent(client_id) or {}
        data = {**intent, "status": status, "updated_at": time.time(), **extra}
        _atomic_write(self.intent_path(client_id), json.dumps(data, ensure_ascii=False, indent=1).encode())
        return data

    def read_intent(self, client_id):
        try:
            return json.loads(self.intent_path(client_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def update(self, prompt_id, status, **extra):
        data = self.read(prompt_id) or {"prompt_id": prompt_id, "at": time.time()}
        data.update(extra, status=status, updated_at=time.time())
        _atomic_write(self.path(prompt_id), json.dumps(data, ensure_ascii=False, indent=1).encode())
        return data

    def read(self, prompt_id):
        try:
            return json.loads(self.path(prompt_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def entries(self):
        if not self.folder.is_dir():
            return []
        out = []
        for path in sorted(self.folder.glob("*.json")):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return out

    def unresolved(self):
        """Prompts whose outcome is still unknown: submitted, or sending when the answer was lost."""
        return [e for e in self.entries() if e.get("status") in (self.SUBMITTED, self.SENDING)]

    def find_done(self, meta):
        """The done record whose metadata matches this cell, or None - the anchor for a download recovery."""
        for entry in self.entries():
            if entry.get("status") != self.DONE or not (entry.get("images") or []):
                continue
            if all(entry.get(key) == value for key, value in meta.items()):
                return entry
        return None


class Comfy:
    def __init__(self, url, journal=None, timeout=1800, poll=2.0, comfy_root=None, opener=None):
        self.url = url.rstrip("/")
        self.journal = Journal(journal) if journal else None
        self.timeout = timeout
        self.poll = poll
        self.comfy_root = str(comfy_root) if comfy_root else None
        self._opener = opener or urllib.request.urlopen
        self._object_info = None

    # ------------------------------------------------------------------ transport
    def _open(self, req, timeout=None):
        return self._opener(req, timeout=self.timeout if timeout is None else timeout)

    def _get(self, path, timeout=None):
        return self._open(urllib.request.Request(self.url + path), timeout).read()

    def _post(self, path, body, timeout=None):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return self._open(req, timeout).read()

    def _json(self, path, timeout=None):
        raw = self._get(path, timeout)
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise ComfyError(f"{path} did not answer with JSON: {raw[:200]!r}") from exc

    # ------------------------------------------------------------------ server facts
    def ping(self, timeout=5):
        try:
            self._get("/system_stats", timeout)
            return True
        except (OSError, ComfyError):
            return False

    def system_stats(self):
        return self._json("/system_stats")

    def object_info(self, node=None):
        """The whole /object_info map (cached) or one node's entry."""
        if self._object_info is None:
            self._object_info = self._json("/object_info", 60)
        if node is None:
            return self._object_info
        if node not in self._object_info:
            raise ComfyError(f"ComfyUI has no node type {node!r}")
        return self._object_info[node]

    def filename_list(self, kind):
        """Every name ComfyUI will accept for this kind of model file, as it lists them."""
        node, widget = FILENAME_LISTS[kind]
        spec = self.object_info(node)["input"]["required"][widget]
        return list(spec[0]) if spec and isinstance(spec[0], list) else []

    def name_for(self, kind, path):
        """The name ComfyUI lists for an absolute model path, or ComfyError naming the folder it must be in.

        The configured paths are absolute so nothing is copied; ComfyUI only accepts names from its own folders, so
        this is where the two are reconciled. A miss re-reads /object_info once: ComfyUI caches that listing, and a
        file dropped into a model folder after the server started can be missing from the first answer.
        """
        if not path:
            raise ComfyError(f"no {kind} path configured")
        try:
            return self._match_name(kind, path)
        except ComfyError:
            self._object_info = None
            return self._match_name(kind, path)

    def _match_name(self, kind, path):
        options = self.filename_list(kind)
        wanted = Path(path)
        roots = _model_roots(self.comfy_root, kind)
        tried = []
        for root in roots:
            try:
                rel = wanted.resolve().relative_to(Path(root).resolve())
            except (ValueError, OSError):
                continue
            for candidate in (str(rel).replace("/", "\\"), str(rel).replace("\\", "/")):
                tried.append(candidate)
                if candidate in options:
                    return candidate
        # Not under a folder ComfyUI reads. The same file may still be reachable through a link or a copy under its
        # own name (the Turbo LoRA ships inside diffusion_models); accept that only when the name is unambiguous.
        same = [o for o in options if o.replace("/", "\\").split("\\")[-1].lower() == wanted.name.lower()]
        if len(same) == 1:
            return same[0]
        if len(same) > 1:
            raise ComfyError(f"{wanted.name} is ambiguous as a {kind}: ComfyUI lists "
                             + ", ".join(sorted(same)) + "; point comfy.root at the right copy")
        raise ComfyError(
            f"{wanted.name} is not visible to ComfyUI as a {kind}. Looked for "
            f"{', '.join(tried) or wanted.name} under {', '.join(str(r) for r in roots)}; "
            f"the file must sit in one of those folders (a hardlink or symlink is enough).")

    def clip_types(self):
        """The CLIPLoader `type` values installed, so doctor can prove krea2 is supported."""
        spec = self.object_info("CLIPLoader")["input"]["required"]["type"]
        return list(spec[0]) if spec and isinstance(spec[0], list) else []

    def missing_nodes(self, classes):
        return [c for c in classes if c not in self.object_info()]

    # ------------------------------------------------------------------ files
    def upload(self, path, subfolder="dataset_pipeline", overwrite=True):
        """Upload an input image; returns the name LoadImage expects."""
        path = Path(path)
        boundary = uuid.uuid4().hex
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parts = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="subfolder"\r\n\r\n{subfolder}\r\n'.encode(),
            (f'--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\n'
             f'{str(bool(overwrite)).lower()}\r\n').encode(),
            (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
             f'filename="{path.parent.name}__{path.name}"\r\nContent-Type: {mime}\r\n\r\n').encode()
            + path.read_bytes() + b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
        req = urllib.request.Request(self.url + "/upload/image", b"".join(parts),
                                     {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        answer = json.loads(self._open(req, 120).read())
        name = answer.get("name") or path.name
        return f"{answer['subfolder']}/{name}" if answer.get("subfolder") else name

    def image_bytes(self, image):
        query = urllib.parse.urlencode({k: image[k] for k in ("filename", "subfolder", "type") if k in image})
        return self._get(f"/view?{query}")

    # ------------------------------------------------------------------ prompts
    def submit(self, graph, **meta):
        """Queue a graph and return its prompt_id, with the id already on disk when this returns.

        The intent is on disk *before* the POST (spec R6): a crash between the request and its answer leaves a record
        that says a submission may be in flight, which `recover` can then settle by client id instead of the caller
        submitting the same graph twice.
        """
        client_id = uuid.uuid4().hex
        if self.journal:
            self.journal.record_intent(client_id, **meta)
        body = json.dumps({"prompt": graph, "client_id": client_id}).encode()
        req = urllib.request.Request(self.url + "/prompt", body, {"Content-Type": "application/json"})
        try:
            answer = json.loads(self._open(req).read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            if self.journal:
                self.journal.drop_intent(client_id, error=f"refused: {detail}")
            raise ComfyError(f"ComfyUI refused the graph: {detail}") from exc
        prompt_id = answer.get("prompt_id")
        if not prompt_id:
            if self.journal:
                self.journal.drop_intent(client_id, error=f"no prompt_id in {answer}")
            raise ComfyError(f"ComfyUI answered without a prompt_id: {answer}")
        if self.journal:
            self.journal.commit_intent(client_id, prompt_id, **meta)
        return prompt_id

    def history(self, prompt_id):
        """The prompt's history entry, or None while it is still queued or running."""
        try:
            return self._json(f"/history/{prompt_id}").get(prompt_id)
        except (OSError, ComfyError):
            return None

    def queue(self):
        return self._json("/queue")

    def queued_ids(self):
        state = self.queue()
        ids = []
        for item in list(state.get("queue_running") or []) + list(state.get("queue_pending") or []):
            if isinstance(item, list) and len(item) > 1:
                ids.append(item[1])
        return ids

    def wait(self, prompt_id, timeout=None, cancel=None):
        """Block until the prompt finishes, returning its outputs. Raises Cancelled if the caller asks to stop."""
        deadline = time.time() + (self.timeout if timeout is None else timeout)
        while True:
            if cancel is not None and cancel():
                self.cancel(prompt_id)
                if self.journal:
                    self.journal.update(prompt_id, Journal.CANCELLED)
                raise Cancelled(f"prompt {prompt_id} cancelled")
            entry = self.history(prompt_id)
            if entry is not None:
                status = entry.get("status") or {}
                if status.get("status_str") != "success":
                    messages = json.dumps(status.get("messages", status))[:2000]
                    if self.journal:
                        self.journal.update(prompt_id, Journal.FAILED, error=messages)
                    raise ComfyError(f"prompt {prompt_id} failed: {messages}")
                outputs = entry.get("outputs") or {}
                if self.journal:
                    self.journal.update(prompt_id, Journal.DONE,
                                        images=_output_images(outputs))
                return outputs
            if time.time() > deadline:
                self.cancel(prompt_id)
                raise ComfyError(f"prompt {prompt_id} produced no result in {timeout or self.timeout}s "
                                 "(ComfyUI restarted or is stuck)")
            time.sleep(self.poll)

    def cancel(self, prompt_id):
        """Take the prompt out of the queue, or interrupt it if it is already running."""
        for path, body in (("/queue", {"delete": [prompt_id]}), ("/interrupt", {"prompt_id": prompt_id})):
            try:
                self._post(path, body)
            except OSError:
                pass

    def run(self, graph, out_path=None, cancel=None, timeout=None, recover=True, **meta):
        """Submit, wait, and return the first output image's bytes (written to `out_path` atomically if given).

        A done journal record for the same cell whose image never landed on disk is a download failure, not a job to
        run again: its output is fetched from the recorded entry instead of submitting the graph a second time
        (spec R6).

        `recover=False` turns that lookup off. It is for a caller whose cells are distinguished by something the
        journal does not record - the holdout, whose two prompts share a seed but not a picture. The recovery matches
        on the caller's `meta` only, so a record from an earlier run with a different prompt but the same slot and seed
        was being adopted as if it were this cell's output; that is how two holdout prompts produced one picture.
        """
        if out_path and self.journal and recover:
            done = self.journal.find_done(meta)
            if done and not Path(out_path).is_file():
                image = (done.get("images") or [None])[0]
                if image:
                    data = self.image_bytes(image)
                    _atomic_write(out_path, data)
                    return data
        prompt_id = self.submit(graph, **meta)
        outputs = self.wait(prompt_id, timeout=timeout, cancel=cancel)
        image = next((i for o in outputs.values() for i in (o.get("images") or [])), None)
        if image is None:
            raise ComfyError(f"prompt {prompt_id} finished without an output image")
        data = self.image_bytes(image)
        if out_path:
            _atomic_write(out_path, data)
        return data

    def _clients_in_flight(self):
        """{client_id: prompt_id} for everything queued or recorded in history, so a lost answer can be adopted."""
        in_flight = {}
        try:
            for item in list(self.queue().get("queue_running") or []) + list(self.queue().get("queue_pending") or []):
                if isinstance(item, list) and len(item) > 3:
                    client_id = (item[3] or {}).get("client_id")
                    if client_id:
                        in_flight[client_id] = item[1]
        except (OSError, ComfyError):
            pass
        try:
            history = json.loads(self._get("/history?max_items=500"))
            for prompt_id, entry in (history or {}).items():
                prompt = entry.get("prompt") or []
                extra = prompt[3] if len(prompt) > 3 else {}
                client_id = (extra or {}).get("client_id")
                if client_id:
                    in_flight.setdefault(client_id, prompt_id)
        except (OSError, ComfyError, ValueError):
            pass
        return in_flight

    def recover(self, cancel=None):
        """Give every journalled prompt whose outcome is unknown a verdict, using /history.

        Returns {prompt_id: entry} for the ones this call settled. A prompt ComfyUI does not know about any more has
        been lost with the server's process, so it is recorded failed rather than left dangling. A *sending* intent -
        the record of a submit whose answer was lost - is settled by client id: adopted when the queue or the history
        still knows it, and recorded failed when the server has no trace at all (spec R6).
        """
        settled = {}
        if not self.journal:
            return settled
        queued = set(self.queued_ids())
        in_flight = self._clients_in_flight()
        for record in self.journal.unresolved():
            prompt_id = record.get("prompt_id")
            if record.get("status") == Journal.SENDING and not prompt_id:
                prompt_id = in_flight.get(record.get("client_id"))
                if not prompt_id:
                    self.journal.drop_intent(record["client_id"], status=Journal.FAILED,
                                             error="unconfirmed: the answer was lost and the server has no trace")
                    settled[f"intent-{record.get('client_id')}"] = None
                    continue
                self.journal.commit_intent(record["client_id"], prompt_id)
            entry = self.history(prompt_id)
            if entry is None:
                if prompt_id in queued:
                    continue
                self.journal.update(prompt_id, Journal.FAILED, error="unknown to ComfyUI after a restart")
                settled[prompt_id] = None
                continue
            status = entry.get("status") or {}
            outputs = entry.get("outputs") or {}
            if status.get("status_str") == "success":
                self.journal.update(prompt_id, Journal.DONE, images=_output_images(outputs))
            else:
                self.journal.update(prompt_id, Journal.FAILED,
                                    error=json.dumps(status.get("messages", status))[:2000])
            settled[prompt_id] = entry
        return settled

    # ------------------------------------------------------------------ GPU
    def free(self, unload_models=True, free_memory=True):
        """Ask ComfyUI to release VRAM. It only unloads models inside its own process; no server is stopped."""
        return self._post("/free", {"unload_models": bool(unload_models),
                                    "free_memory": bool(free_memory)})

    def idle(self):
        """True when nothing is queued or running."""
        state = self.queue()
        return not state.get("queue_running") and not state.get("queue_pending")


def _output_images(outputs):
    return [dict(filename=i.get("filename"), subfolder=i.get("subfolder"), type=i.get("type"))
            for o in outputs.values() for i in (o.get("images") or [])]


def _model_roots(comfy_root, kind):
    if not comfy_root:
        raise ComfyError("comfy.root is not configured, so model filenames cannot be resolved")
    root = Path(comfy_root) / "models"
    if kind == "clip":
        return [root / "text_encoders", root / "clip"]
    return {"unet": [root / "diffusion_models"], "lora": [root / "loras"], "vae": [root / "vae"]}[kind]


def save_image(path, data):
    """Public form of the atomic write, for callers that already have the bytes."""
    return _atomic_write(path, data)
