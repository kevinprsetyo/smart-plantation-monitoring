"""
main.py
=======
Entry point FastAPI untuk Smart Plantation Safety Monitoring System.

Endpoint yang tersedia:
    GET  /                              : Health check dan info sistem.
    GET  /video/{filename}              : Serve file video output (H.264 MP4).
    GET  /gallery                       : Daftar semua snapshot pelanggaran.
    GET  /snapshot/{category}/{filename}: Serve satu file snapshot JPEG.
    POST /predict-image                 : Deteksi PPE pada gambar statis.
    POST /predict-video                 : Deteksi PPE pada video (frame by frame).

Catatan path:
    Seluruh direktori menggunakan path absolut yang diresolvsi relatif
    terhadap file ini (bukan working directory saat server dijalankan)
    untuk menghindari masalah kompatibilitas path di Windows.

Cara menjalankan:
    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
import cv2
import numpy as np
from pathlib import Path
import shutil

from backend.detector import PlantationDetector
from backend.tracker import PlantationTracker
from backend.violation_engine import ViolationEngine
from backend.video_processor import VideoProcessor
from backend.ai_report_generator import generate_report, generate_image_report


# ---------------------------------------------------------------------------
# INISIALISASI APLIKASI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Smart Plantation Safety Monitoring",
    description="API deteksi pelanggaran keselamatan kerja di perkebunan menggunakan YOLOv8.",
    version="1.0.0"
)

# Komponen pipeline (diinisialisasi sekali saat server start)
detector        = PlantationDetector()
tracker         = PlantationTracker()
violation_engine = ViolationEngine()
video_processor  = VideoProcessor()

# Direktori absolut berdasarkan lokasi file main.py (backend/)
BASE_DIR      = Path(__file__).resolve().parent
UPLOAD_DIR    = BASE_DIR / "uploads"
OUTPUT_DIR    = BASE_DIR / "outputs"
SNAPSHOT_DIR  = BASE_DIR / "snapshots"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR.mkdir(exist_ok=True)

# Peta nama subdirektori snapshot ke label kategori yang ditampilkan
SNAPSHOT_FOLDER_MAP = {
    "no_helmet":      "NO_HELMET",
    "no_safety_vest": "NO_SAFETY_VEST",
    "fire":           "FIRE_ALERT",
    "smoke":          "SMOKE_ALERT"
}

# Validasi kategori yang diizinkan di endpoint /snapshot
VALID_SNAPSHOT_CATEGORIES = set(SNAPSHOT_FOLDER_MAP.keys())


# ---------------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    """
    Health check endpoint.

    Returns:
        dict: Pesan status dan versi sistem.
    """
    return {
        "message": "Smart Plantation Safety Monitoring System",
        "status":  "running"
    }


# ---------------------------------------------------------------------------
# VIDEO SERVE ENDPOINT
# ---------------------------------------------------------------------------

@app.get("/video/{filename}")
def serve_video(filename: str):
    """
    Serve file video output yang sudah diproses (H.264 MP4).

    Endpoint ini digunakan oleh Streamlit untuk mengambil video sebagai
    bytes melalui HTTP, menghindari masalah perbedaan working directory
    antara proses FastAPI dan proses Streamlit.

    Args:
        filename (str): Nama file video di direktori outputs/.

    Returns:
        FileResponse: Stream file MP4 dengan media type video/mp4.

    Raises:
        HTTPException (404): Jika file tidak ditemukan di outputs/.
    """
    video_path = OUTPUT_DIR / filename

    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Video tidak ditemukan: {filename}"
        )

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=filename
    )


# ---------------------------------------------------------------------------
# VIOLATION GALLERY ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/gallery")
def get_gallery():
    """
    Kembalikan daftar semua snapshot pelanggaran yang tersimpan.

    Membaca direktori snapshots/ dan mengelompokkan file berdasarkan
    jenis pelanggaran. File diurutkan secara kronologis berdasarkan
    nama file (yang mengandung timestamp).

    Returns:
        dict: Snapshot dikelompokkan per kategori pelanggaran.
            Struktur:
            {
                "NO_HELMET": {
                    "folder": "no_helmet",
                    "files": ["20260623_103000_000001.jpg", ...]
                },
                "NO_SAFETY_VEST": {...},
                "FIRE_ALERT": {...},
                "SMOKE_ALERT": {...}
            }
    """
    result = {}

    for folder_name, category_label in SNAPSHOT_FOLDER_MAP.items():

        folder_path = SNAPSHOT_DIR / folder_name

        if folder_path.exists():
            # Ambil semua file JPEG, urutkan kronologis berdasarkan nama file
            files = sorted([
                f.name
                for f in folder_path.iterdir()
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
            ])
        else:
            files = []

        result[category_label] = {
            "folder": folder_name,
            "files":  files
        }

    return result


@app.get("/snapshot/{category}/{filename}")
def serve_snapshot(category: str, filename: str):
    """
    Serve satu file snapshot JPEG dari direktori snapshots/.

    Digunakan oleh Streamlit untuk menampilkan preview dan menyediakan
    tombol download per gambar di Violation Gallery.

    Args:
        category (str): Nama subdirektori (salah satu dari: no_helmet,
                        no_safety_vest, fire, smoke).
        filename (str): Nama file gambar di dalam subdirektori tersebut.

    Returns:
        FileResponse: File JPEG dengan media type image/jpeg.

    Raises:
        HTTPException (400): Jika category tidak valid.
        HTTPException (404): Jika file tidak ditemukan.
    """
    # Validasi kategori untuk mencegah path traversal attack
    if category not in VALID_SNAPSHOT_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Kategori tidak valid: '{category}'. "
                   f"Pilihan: {sorted(VALID_SNAPSHOT_CATEGORIES)}"
        )

    img_path = SNAPSHOT_DIR / category / filename

    if not img_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Snapshot tidak ditemukan: {category}/{filename}"
        )

    return FileResponse(
        path=str(img_path),
        media_type="image/jpeg",
        filename=filename
    )


# ---------------------------------------------------------------------------
# IMAGE DETECTION
# ---------------------------------------------------------------------------

@app.post("/predict-image")
async def predict_image(file: UploadFile):
    """
    Deteksi PPE pada gambar statis.

    Langkah-langkah:
        1. Decode bytes gambar ke array OpenCV.
        2. Jalankan deteksi YOLOv8.
        3. Update tracker ByteTrack.
        4. Periksa pelanggaran PPE.
        5. Gambar bounding box dan label.
        6. Simpan gambar output ke outputs/output_image.jpg.

    Args:
        file (UploadFile): File gambar (JPG/PNG) dari form upload.

    Returns:
        dict: Hasil deteksi dengan struktur:
            {
                "output_image": str,
                "ppe_statistics": {
                    "safe_worker": int,
                    "no_helmet": int,
                    "no_safety_vest": int
                },
                "violations": list[dict],
                "object_count": dict
            }

    Raises:
        HTTPException (400): Jika file bukan gambar yang valid.
    """
    contents = await file.read()

    image = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="File bukan gambar yang valid."
        )

    # Pipeline: deteksi -> tracking -> violation check
    detections = detector.detect(image)
    tracks     = tracker.update(detections)
    violations = violation_engine.check(detections, tracks)

    # Buat peta track_id -> status untuk efisiensi lookup
    violation_map = {
        v["track_id"]: v["status"]
        for v in violations
        if "track_id" in v
    }

    # -----------------------------------------------------------------------
    # GAMBAR BOUNDING BOX DAN LABEL
    # -----------------------------------------------------------------------
    for track in tracks:

        x1, y1, x2, y2 = track["bbox"]
        track_id = track["track_id"]
        status   = violation_map.get(track_id, "SAFE")

        color = (0, 255, 0)              # hijau  = SAFE
        if status == "NO_HELMET":
            color = (0, 0, 255)          # merah  = NO_HELMET
        elif status == "NO_SAFETY_VEST":
            color = (0, 165, 255)        # oranye = NO_SAFETY_VEST

        label = f"Worker #{track_id} | {status}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    # -----------------------------------------------------------------------
    # SIMPAN GAMBAR OUTPUT
    # -----------------------------------------------------------------------
    output_path = OUTPUT_DIR / "output_image.jpg"
    cv2.imwrite(str(output_path), image)

    # -----------------------------------------------------------------------
    # HITUNG PPE STATISTICS
    # -----------------------------------------------------------------------
    safe_worker    = sum(1 for v in violations if v["status"] == "SAFE")
    no_helmet      = sum(1 for v in violations if v["status"] == "NO_HELMET")
    no_safety_vest = sum(1 for v in violations if v["status"] == "NO_SAFETY_VEST")

    # -----------------------------------------------------------------------
    # HITUNG JUMLAH OBJEK PER KELAS
    # -----------------------------------------------------------------------
    object_count = {
        "person":      len([d for d in detections if d["class_name"] == "person"]),
        "helmet":      len([d for d in detections if d["class_name"] == "helmet"]),
        "safety_vest": len([d for d in detections if d["class_name"] == "safety_vest"]),
        "truck":       len([d for d in detections if d["class_name"] == "truck"]),
        "forklift":    len([d for d in detections if d["class_name"] == "forklift"]),
        "excavator":   len([d for d in detections if d["class_name"] == "excavator"]),
        "fire":        len([d for d in detections if d["class_name"] == "fire"]),
        "smoke":       len([d for d in detections if d["class_name"] == "smoke"])
    }

    # -----------------------------------------------------------------------
    # HASILKAN AI SAFETY REPORT UNTUK GAMBAR
    # generate_image_report() tidak pernah melempar exception ke sini.
    # -----------------------------------------------------------------------
    ai_report = generate_image_report({
        "safe_worker":    safe_worker,
        "no_helmet":      no_helmet,
        "no_safety_vest": no_safety_vest,
        "object_count":   object_count
    })

    return {
        "output_image": str(output_path),
        "ppe_statistics": {
            "safe_worker":    safe_worker,
            "no_helmet":      no_helmet,
            "no_safety_vest": no_safety_vest
        },
        "violations":   violations,
        "object_count": object_count,
        "ai_report":    ai_report
    }


# ---------------------------------------------------------------------------
# VIDEO DETECTION
# ---------------------------------------------------------------------------

@app.post("/predict-video")
async def predict_video(file: UploadFile):
    """
    Deteksi PPE pada video (frame by frame).

    Langkah-langkah:
        1. Validasi ekstensi file (.mp4).
        2. Simpan file upload ke uploads/.
        3. Jalankan VideoProcessor.process_video().
        4. Hasilkan AI Safety Report menggunakan Ollama Cloud API.
        5. Kembalikan nama file output, statistik, timeline, dan laporan AI.

    Video output disimpan di outputs/ dalam format H.264 MP4 dan dapat
    diakses melalui endpoint GET /video/{filename}.

    Args:
        file (UploadFile): File video MP4 dari form upload.

    Returns:
        dict: Hasil pemrosesan dengan struktur:
            {
                "message"         : str,
                "input_video"     : str,
                "output_video"    : str,
                "output_video_url": str,
                "statistics"      : dict,
                "violations"      : list[dict],
                "object_count"    : dict,
                "timeline_events" : list[dict],  # event kronologis
                "ai_report"       : str          # laporan dari LLM
            }

    Raises:
        HTTPException (400): Jika file bukan MP4 atau file kosong.
        HTTPException (500): Jika proses video gagal.
    """
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=400,
            detail="Hanya file MP4 yang didukung."
        )

    # Sanitasi nama file untuk mencegah path traversal attack
    safe_filename   = Path(file.filename).name
    input_path      = UPLOAD_DIR / safe_filename
    output_filename = "output_" + safe_filename
    output_path     = OUTPUT_DIR / output_filename

    # Simpan file upload ke disk
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if input_path.stat().st_size == 0:
        raise HTTPException(
            status_code=400,
            detail="File video kosong."
        )

    # -----------------------------------------------------------------------
    # PROSES VIDEO
    # -----------------------------------------------------------------------
    try:
        result = video_processor.process_video(
            str(input_path),
            str(output_path)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Video processing gagal: {str(e)}"
        )

    # -----------------------------------------------------------------------
    # HASILKAN AI SAFETY REPORT
    # generate_report() tidak pernah melempar exception ke sini.
    # -----------------------------------------------------------------------
    ai_report = generate_report(result["statistics"])

    return {
        "message":          "Video processing selesai",
        "input_video":      safe_filename,
        "output_video":     output_filename,
        "output_video_url": f"/video/{output_filename}",
        "statistics":       result["statistics"],
        "violations":       result["violations"],
        "object_count":     result["object_count"],
        "timeline_events":  result["timeline_events"],
        "ai_report":        ai_report
    }