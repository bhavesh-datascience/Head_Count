import base64
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from detector import PersonDetector
from models.runtime import configure_threads
from routes.unique_count import router as unique_count_router
from services.unique_count_service import shutdown_runtime, start_runtime

BASE_DIR = Path(__file__).parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

detector: PersonDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    # Size PyTorch/OpenCV thread pools to the CPUs actually available before
    # any model runs (matters on small CPU servers and in containers).
    configure_threads()
    detector = PersonDetector()
    # Shared with /unique-count so the YOLO model is loaded only once.
    app.state.person_detector = detector
    # Starts the /unique-count GPU worker and warms its models up in the background.
    start_runtime(detector)
    yield
    shutdown_runtime()


app = FastAPI(title="Person Counter", version="1.0.0", lifespan=lifespan)
app.include_router(unique_count_router)


def annotate(image: np.ndarray, detections: list[dict]) -> str:
    out = image.copy()
    thickness = max(2, round(max(out.shape[:2]) / 600))
    for n, d in enumerate(detections, 1):
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), thickness)
        cv2.putText(out, str(n), (x1 + 3, y1 + 18 * thickness // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thickness, (0, 0, 255), thickness)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@app.get("/", response_class=HTMLResponse)
async def index():
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(detector.device) if detector else None}


async def read_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image larger than 20 MB")

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image")
    return image


@app.post("/number-count")
async def number_count(
    file: UploadFile = File(..., description="Image file (jpg, png, webp, bmp...)"),
    tiles: bool = Query(True, description="Extra sliced pass for small / crowded people"),
):
    image = await read_image(file)
    detections = await run_in_threadpool(detector.detect, image, tiles)
    return {"total_persons": len(detections)}


@app.post("/count")
async def count_persons(
    file: UploadFile = File(..., description="Image file (jpg, png, webp, bmp...)"),
    tiles: bool = Query(True, description="Extra sliced pass for small / crowded people"),
    annotated: bool = Query(False, description="Return the image with boxes as base64"),
):
    image = await read_image(file)
    detections = await run_in_threadpool(detector.detect, image, tiles)
    h, w = image.shape[:2]
    result = {
        "total_persons": len(detections),
        "image_width": w,
        "image_height": h,
        "detections": detections,
    }
    if annotated:
        result["annotated_image"] = annotate(image, detections)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001)
