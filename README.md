# Person Count

A FastAPI service that counts people in photos, with a web UI.

| Endpoint | What it does |
|---|---|
| `POST /count` | Counts people in **one** image (YOLO), optionally returns an annotated image |
| `POST /number-count` | Same as `/count`, returns only the number |
| `POST /unique-count` | Counts **unique** people across **many** images (the same person appearing in several photos is counted once) |
| `GET /` | Web UI with a single-image tab and a multi-image "unique count" tab |

`/unique-count` uses InsightFace (`buffalo_l`: SCRFD face detection and ArcFace embeddings) as the main identity signal. OSNet person re-identification, run on YOLO person crops, is supporting evidence only and never overrides strong face evidence. Faces are grouped by agglomerative clustering, and two faces in the same image can never be merged. Pairs the service isn't sure about are listed for manual review rather than merged.

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11 or Linux, 64-bit | |
| Python | 3.13 | |
| GPU | None (runs on CPU) | NVIDIA RTX 20-series or newer, **8 GB VRAM** (12 GB+ for comfortable headroom) |
| NVIDIA driver | — | Version that supports CUDA 13 (580 or newer) |
| Disk | ~2 GB (CPU install) / ~6 GB (GPU install, includes CUDA libraries) | |
| RAM | 4 GB | 8 GB |

You don't need to install the CUDA Toolkit or cuDNN yourself. They are installed as Python packages. You only need an up-to-date NVIDIA driver.

**No GPU?** The service switches to the CPU automatically and uses lighter CPU settings (see [Deploying on a CPU server](#deploying-on-a-cpu-server-aws-t3large--t3xlarge)). On 2 CPU cores `/unique-count` takes about 3 s per image and uses about 1.6 GB of RAM at peak. AMD and Intel GPUs are not supported and are treated as "no GPU".

Measured VRAM on an RTX 5070 Ti:

| Situation | VRAM |
|---|---|
| Models loaded, idle | ~1.9 GB |
| Typical photos (1–2 MP) | ~3 GB peak |
| Large photos (12 MP) | ~4.3 GB peak (temporary) |

---

## Installation

### Option A — uv (recommended)

[uv](https://docs.astral.sh/uv/) reads `pyproject.toml` and `uv.lock`, which already pick the CUDA builds of PyTorch and ONNX Runtime.

```bash
# 1. Install uv (once)
#    Windows (PowerShell):  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
#    Linux / macOS:         curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Get the code
git clone <your-repo-url> person_count
cd person_count

# 3. Create .venv with Python 3.13 and install every dependency
uv sync
```

### Option B — pip

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux: source .venv/bin/activate

# PyTorch with CUDA 13 (on a CPU-only machine, drop --index-url)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements.txt

# insightface pulls in the CPU-only "onnxruntime", which would overwrite the
# GPU build. Remove it and reinstall the GPU build:
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-gpu
```

On a machine without an NVIDIA GPU, use `pip install -r requirements-cpu.txt` instead. It installs the much smaller CPU builds (see [Deploying on a CPU server](#deploying-on-a-cpu-server-aws-t3large--t3xlarge)).

### Model files

These download automatically on first start:

| Model | Location | Size |
|---|---|---|
| YOLO11m (person detection) | `yolo11m.pt` (included in the repo) | 40 MB |
| InsightFace `buffalo_l` | `~/.insightface/models/buffalo_l/` | ~330 MB |
| OSNet x1_0 (MSMT17) | `weights/osnet_x1_0_msmt17.pth` | 17 MB |

The first start therefore needs internet access. After that the service works offline. To get rid of the leftover download, delete `~/.insightface/models/buffalo_l.zip` (~276 MB).

---

## Running

```bash
# uv
uv run python main.py
#   or: uv run uvicorn main:app --host 0.0.0.0 --port 8001

# pip (with the venv activated)
python main.py
```

Then open **http://localhost:8001**. Interactive API docs are at **http://localhost:8001/docs**.

At startup the models load and warm up in the background, which takes a few seconds on a GPU. To check that everything is on the GPU:

```bash
curl http://localhost:8001/unique-count/status
```

Look at `"devices"`: each model shows either `cuda` or `cpu`, and `"profile"` shows `gpu` or `cpu`. `GET /health` shows the device that `/count` runs on.

---

## Deploying on a CPU server (AWS t3.large / t3.xlarge)

The service runs on a CPU-only machine with 2 vCPUs and 8 GB RAM (t3.large) or more (t3.xlarge: 4 vCPUs, 16 GB). Without a GPU it switches automatically to a **CPU profile**:

- Faces are detected at one size, 960 px, instead of two (640 + 1280).
- Images are downscaled to at most 2560 px.
- No extra tiled person-detection pass in `/unique-count`.
- Small batches.
- Thread pools sized to the CPUs actually available.

Every individual setting can still be overridden; see [Configuration](#configuration).

Measured with CPU only on 2 cores (the same images and results before and after):

| | Before optimisation | Now |
|---|---|---|
| Startup (models ready) | 76 s | 5 s |
| RAM when idle | 2.1 GB | 1.0 GB |
| RAM peak (12 MP photos) | 6.3 GB | 1.6 GB |
| `/unique-count`, 3 normal photos | 243 s | 8.6 s |
| `/unique-count`, 2 × 12 MP + 1 photo | 312 s | 9.6 s |
| `/count`, one 12 MP photo | 15 s | 15 s |

### Install (Ubuntu)

```bash
# System libraries needed by OpenCV
sudo apt update && sudo apt install -y git libgl1 libglib2.0-0

# uv (also provides Python 3.13 if the system Python is older)
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

git clone https://github.com/bhavesh-datascience/Head_Count.git /var/www/facecount
cd /var/www/facecount
uv venv --python 3.13
# CPU builds of PyTorch / ONNX Runtime: about 1 GB instead of about 6 GB of CUDA libraries
uv pip install -r requirements-cpu.txt --no-config --index-strategy unsafe-best-match
```

Plain pip works too: `python3.13 -m venv .venv && .venv/bin/pip install -r requirements-cpu.txt`.

Don't run `uv sync` on a CPU server: it installs the GPU builds from `pyproject.toml`.

### Run as a service

[`deploy/facecount.service`](deploy/facecount.service) is a ready-made systemd unit. It does the following:

- runs one worker
- sets the CPU profile and the memory-saving environment variables
- caps the service at 7 GB with `MemoryMax`, so it can never take the whole machine down
- restarts the service if it crashes

```bash
sudo cp deploy/facecount.service /etc/systemd/system/facecount.service
# edit User / WorkingDirectory / port in the file if yours differ
sudo systemctl daemon-reload && sudo systemctl enable --now facecount
journalctl -u facecount -f          # wait for "All /unique-count models ready"
```

### Tips for t3 instances

- t3 instances are **burstable**: once their CPU credits run out they're throttled to 20% CPU. For steady traffic, turn on **T3 Unlimited** or use a non-burstable type such as c7i.large or c7i.xlarge.
- Use a disk of at least 20 GB.
- Use `async_mode=true` for `/unique-count` with many images, so proxies don't time out.
- `/count` is the slowest call on CPU (about 15 s for a 12 MP photo). Each large photo is scanned in overlapping tiles to find small, distant people. For faster responses, send `tiles=false` with each request, or set `PERSON_MODEL=yolo11s.pt` (a smaller model that finds fewer small people).

---

## API usage

### Single image

```bash
curl -F "file=@photo.jpg" "http://localhost:8001/count?annotated=false"
```

### Unique people across several images

```bash
curl -F "files=@img1.jpg" -F "files=@img2.png" -F "files=@img3.webp" \
     "http://localhost:8001/unique-count"
```

Accepted formats are JPG, JPEG, PNG and WEBP: up to 20 MB per file and 500 files per request.

| Query parameter | Default | Meaning |
|---|---|---|
| `skip_invalid` | `false` | Skip corrupted images instead of failing the whole request |
| `include_details` | `false` | Include per-face boxes and scores for each identity |
| `async_mode` | `false` | Return `202` with a `job_id` immediately, then poll `GET /unique-count/jobs/{job_id}` |

Example response (shortened):

```json
{
  "success": true,
  "processing_status": "completed",
  "total_images": 3,
  "processed_images": 3,
  "failed_images": 0,
  "total_detected_faces": 13,
  "unique_person_count": 7,
  "identities": [
    { "person_id": "PERSON_001", "appearance_count": 2, "image_ids": ["IMAGE_001", "IMAGE_002"],
      "confidence": 0.91, "match_basis": "face" }
  ],
  "uncertain_matches": [],
  "devices": { "face_detector": "cuda", "face_recognizer": "cuda", "osnet": "cuda:0 (fp16)", "person_detector": "cuda:0" }
}
```

Errors always have the same shape:

```json
{ "success": false, "processing_status": "failed",
  "error": { "code": "UNSUPPORTED_FORMAT", "message": "...", "details": {} } }
```

Status codes:

| Code | Meaning |
|---|---|
| 400 | No images |
| 413 | Too many images, or a file is too large |
| 415 | Wrong format |
| 422 | Corrupted image (without `skip_invalid`) |
| 429 | Queue full |
| 503 | Models failed to load |

---

## Configuration

All settings are optional environment variables.

| Variable | Default | Description |
|---|---|---|
| `UC_PROFILE` | `auto` | `gpu` or `cpu` defaults. `auto` picks `cpu` when no CUDA GPU is available. |
| `UC_CPU_THREADS` | `0` (auto) | Threads per inference library on CPU. Auto means the CPUs the process can use. |
| `UC_ORT_CPU_ARENA` | GPU: `true`, CPU: `false` | Memory pool for the face models on CPU: slightly faster, but keeps peak memory resident |
| `UC_REQUIRE_GPU` | `false` | Fail (503) instead of silently running on the CPU |
| `UC_GPU_DEVICE` | `0` | GPU index |
| `UC_ORT_GPU_MEM_MB` | `0` (no limit) | VRAM cap for the face models |
| `UC_OSNET_FP16` | `true` | Run Re-ID in half precision on the GPU |
| `UC_PRELOAD_MODELS` | `true` | Load the models at startup instead of on the first request |
| `UC_MAX_CONCURRENT_JOBS` | `1` | Jobs processed at the same time on the GPU |
| `UC_MAX_PENDING_JOBS` | `8` | Queue size; further requests get a 429 |
| `UC_JOB_TTL_SECONDS` | `3600` | How long finished job results stay in memory |
| `UC_MAX_IMAGES` | `500` | Maximum images per request |
| `UC_MAX_UPLOAD_MB` | `20` | Maximum size per image |
| `UC_MAX_IMAGE_SIDE` | GPU: `4096`, CPU: `2560` | Larger images are downscaled to this |
| `UC_UPLOAD_DIR` | system temp | Where uploads are kept while a job runs |
| `UC_DET_SIZES` | GPU: `640,1280`, CPU: `960` | Face-detection input sizes (comma-separated for multi-scale) |
| `UC_DET_THRESH` | `0.50` | Minimum face-detection score |
| `UC_MIN_FACE_SIZE` | `20` | Minimum face size in pixels |
| `UC_FACE_STRONG` | `0.55` | Cosine similarity that counts as the same person |
| `UC_FACE_WEAK` | `0.30` | Below this, faces are treated as different people |
| `UC_CLUSTER_THRESHOLD` | `0.42` | Clustering merge threshold |
| `UC_REID_ENABLED` | `true` | Use OSNet body Re-ID as supporting evidence |
| `UC_REID_BATCH` | GPU: `32`, CPU: `8` | Person crops per Re-ID batch |
| `UC_PERSON_TILES` | GPU: `true`, CPU: `false` | Extra tiled person-detection pass in `/unique-count` |
| `UC_COUNT_BODY_ONLY` | `false` | Count people whose face is never visible |
| `UC_FACE_MODEL_ROOT` | `~/.insightface` | Where InsightFace models are stored |
| `UC_OSNET_WEIGHTS` | `weights/osnet_x1_0_msmt17.pth` | OSNet weights path |
| `PERSON_MODEL` | `yolo11m.pt` | YOLO model for person detection (`yolo11s.pt` / `yolo11n.pt` are lighter; they download automatically) |
| `PERSON_IMGSZ` | `1280` | YOLO input size for the full-image pass |
| `PERSON_TILES` | `true` | Default for `/count`'s `tiles` parameter |
| `PERSON_TILE_BATCH` | GPU: `16`, CPU: `2` | Tiles per YOLO batch. Lower uses less memory, and results are the same. |

Example (PowerShell):

```powershell
$env:UC_REQUIRE_GPU = "true"; uv run python main.py
```

---

## Privacy and storage

- Nothing is stored permanently: there is no database, and no images or face embeddings are saved.
- Uploads are written to a temporary job folder. Each file is deleted as soon as it has been read, and the folder is removed when the job ends.
- Finished job results (person IDs, image IDs and filenames, but no images or embeddings) stay in memory for `UC_JOB_TTL_SECONDS` and are lost on restart.
- Each request is independent: people are not remembered between requests.

---

## Project structure

```
main.py                   FastAPI app, /count, /number-count, web UI
detector.py               YOLO person detector (shared, thread-safe)
routes/unique_count.py    /unique-count HTTP endpoints
services/                 pipeline: config, jobs/queue, model lifecycle,
                          face detection/recognition, Re-ID, matching
models/                   InsightFace and OSNet wrappers, GPU and CPU/thread setup
utils/                    image decoding/validation, similarity helpers
templates/index.html      web UI
deploy/facecount.service  systemd unit for a CPU server
requirements-cpu.txt      CPU-only dependencies (servers without a GPU)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `devices` shows `cpu` although you have an NVIDIA GPU | Update the NVIDIA driver (CUDA 13 support needed). With pip, make sure the CPU-only `onnxruntime` is uninstalled (see Option B). |
| `uv sync` fails with "Access is denied" on a `.dll` (Windows) | The server is still running and holding the file. Stop it and run `uv sync` again. |
| First request is slow | The models are still loading or downloading. Check `GET /unique-count/status`. |
| Out of GPU memory on very large photos | Lower `UC_MAX_IMAGE_SIDE` (for example to `2560`) or use a GPU with more VRAM. |
| 429 `QUEUE_FULL` | Too many jobs at once. Retry later or raise `UC_MAX_PENDING_JOBS`. |
| `oom-kill` in `journalctl` on a CPU server | Make sure the CPU install was used (`pip list` must not show `+cu130` or `nvidia-*` packages), run only one worker, and use `deploy/facecount.service`. On 4 GB machines also add swap. |
| `ImportError: libGL.so.1` | `sudo apt install -y libgl1 libglib2.0-0` |
| Very slow on a t3 after a while | CPU credits are used up. Turn on T3 Unlimited or use a non-burstable instance. |

## Limitations

- Identity relies mainly on faces. People who never show their face are not counted unless `UC_COUNT_BODY_ONLY=true`, and that mode is less reliable.
- Clothing similarity alone never merges two people. Twins or very similar-looking people may need the manual review list.
- Results are estimates. Check `uncertain_matches` when accuracy matters.
