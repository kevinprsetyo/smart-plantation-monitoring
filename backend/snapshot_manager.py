"""
snapshot_manager.py
===================
Modul penyimpanan snapshot frame yang mengandung pelanggaran PPE
atau deteksi bahaya kebakaran/asap.

Snapshot disimpan dalam direktori terstruktur berdasarkan jenis pelanggaran:
    backend/snapshots/
        no_helmet/       -> frame dengan worker tanpa helm
        no_safety_vest/  -> frame dengan worker tanpa rompi keselamatan
        fire/            -> frame dengan deteksi api
        smoke/           -> frame dengan deteksi asap

Nama file menggunakan timestamp dengan presisi mikrodetik untuk menghindari
tabrakan nama file ketika banyak pelanggaran terjadi dalam waktu berdekatan.

Format nama file: YYYYMMDD_HHMMSS_ffffff.jpg
"""

from pathlib import Path
from datetime import datetime
import cv2


class SnapshotManager:
    """
    Manajer penyimpanan snapshot frame pelanggaran.

    Membuat direktori kategori pelanggaran secara otomatis saat
    diinisialisasi dan menyimpan frame ke direktori yang sesuai.

    Attributes:
        base_dir (Path): Direktori root penyimpanan snapshot.
    """

    # Peta status pelanggaran ke nama subdirektori
    VIOLATION_DIR_MAP = {
        "NO_HELMET":      "no_helmet",
        "NO_SAFETY_VEST": "no_safety_vest",
        "FIRE_ALERT":     "fire",
        "SMOKE_ALERT":    "smoke"
    }

    def __init__(self):
        """
        Inisialisasi direktori penyimpanan snapshot.

        Membuat semua subdirektori kategori pelanggaran jika belum ada.
        """
        self.base_dir = Path("backend/snapshots")

        for subdir in self.VIOLATION_DIR_MAP.values():
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

    def save(self, frame, violation_type: str) -> str:
        """
        Simpan frame sebagai snapshot JPEG ke direktori kategori pelanggaran.

        Nama file dibuat menggunakan timestamp saat ini dengan presisi
        mikrodetik untuk memastikan keunikan nama.

        Args:
            frame (numpy.ndarray): Frame gambar BGR dari OpenCV yang akan disimpan.
            violation_type (str) : Jenis pelanggaran sebagai kunci pemetaan direktori.
                Nilai yang valid: 'NO_HELMET', 'NO_SAFETY_VEST', 'FIRE_ALERT', 'SMOKE_ALERT'.

        Returns:
            str: Path absolut file snapshot yang disimpan, atau string kosong
                 jika violation_type tidak dikenali.
        """
        folder = self.VIOLATION_DIR_MAP.get(violation_type)

        if folder is None:
            return ""

        filename  = datetime.now().strftime("%Y%m%d_%H%M%S_%f.jpg")
        save_path = self.base_dir / folder / filename

        cv2.imwrite(str(save_path), frame)

        return str(save_path)