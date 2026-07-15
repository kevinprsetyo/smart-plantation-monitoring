"""
violation_engine.py
===================
Modul pemeriksaan pelanggaran PPE (Personal Protective Equipment) dan
deteksi bahaya kebakaran/asap.

Logika pemeriksaan PPE:
    Setiap person yang sedang dilacak diperiksa apakah memiliki:
        1. Helm (helmet): Pusat bbox helm harus berada di dalam 35% atas
           bbox person (area kepala).
        2. Rompi keselamatan (safety_vest): Pusat bbox rompi harus berada
           di dalam 25%-100% bbox person (area torso ke bawah).

    Prioritas status (jika keduanya tidak ada, lapor NO_HELMET lebih dulu):
        SAFE          -> helmet ada DAN rompi ada
        NO_HELMET     -> helmet tidak ada (rompi diabaikan)
        NO_SAFETY_VEST -> helmet ada TAPI rompi tidak ada

Status yang mungkin dihasilkan:
    "SAFE"           : Worker lengkap PPE-nya.
    "NO_HELMET"      : Worker tidak terdeteksi memakai helm.
    "NO_SAFETY_VEST" : Worker terdeteksi memakai helm tapi tidak rompi.
    "FIRE_ALERT"     : Terdeteksi api dalam frame (tidak terkait person).
    "SMOKE_ALERT"    : Terdeteksi asap dalam frame (tidak terkait person).
"""


class ViolationEngine:
    """
    Mesin pemeriksaan pelanggaran PPE dan bahaya lingkungan.

    Memeriksa setiap worker (person yang sedang dilacak) terhadap
    keberadaan helm dan rompi keselamatan menggunakan analisis
    spatial overlap antara bounding box objek.
    """

    def __init__(self):
        """Inisialisasi ViolationEngine (tidak membutuhkan state internal)."""
        pass

    def _is_center_inside_region(
        self,
        obj_bbox: list,
        ref_bbox: list,
        y_min_ratio: float = 0.0,
        y_max_ratio: float = 1.0
    ) -> bool:
        """
        Periksa apakah pusat suatu objek berada di dalam region tertentu
        dari bounding box referensi.

        Args:
            obj_bbox (list)    : [x1, y1, x2, y2] objek yang diperiksa (helm/rompi).
            ref_bbox (list)    : [x1, y1, x2, y2] objek referensi (person).
            y_min_ratio (float): Batas atas region (proporsi tinggi ref_bbox).
            y_max_ratio (float): Batas bawah region (proporsi tinggi ref_bbox).

        Returns:
            bool: True jika pusat objek berada di dalam region yang ditentukan.
        """
        ox1, oy1, ox2, oy2 = obj_bbox
        px1, py1, px2, py2 = ref_bbox

        cx = (ox1 + ox2) / 2
        cy = (oy1 + oy2) / 2

        person_height = py2 - py1

        region_top    = py1 + person_height * y_min_ratio
        region_bottom = py1 + person_height * y_max_ratio

        return (px1 <= cx <= px2) and (region_top <= cy <= region_bottom)

    def check(self, detections: list, tracks: list) -> list:
        """
        Periksa pelanggaran PPE dan deteksi bahaya untuk semua track aktif.

        Untuk setiap person yang sedang dilacak:
            - Cari helm yang pusatnya berada di 35% atas bbox person.
            - Cari rompi yang pusatnya berada di 25%-100% bbox person.
            - Tentukan status berdasarkan keberadaan masing-masing PPE.

        Untuk deteksi api/asap:
            - Jika ada minimal satu objek kelas 'fire', tambahkan FIRE_ALERT.
            - Jika ada minimal satu objek kelas 'smoke', tambahkan SMOKE_ALERT.

        Args:
            detections (list[dict]): Output dari PlantationDetector.detect().
            tracks (list[dict])    : Output dari PlantationTracker.update().

        Returns:
            list[dict]: Daftar pelanggaran. Setiap item memiliki struktur:
                - Untuk worker:
                    {"track_id": int, "status": str}
                - Untuk kebakaran/asap (tidak ada track_id):
                    {"status": "FIRE_ALERT"} atau {"status": "SMOKE_ALERT"}
        """
        violations = []

        # Kelompokkan deteksi berdasarkan kelas untuk efisiensi lookup
        helmets = [d for d in detections if d["class_name"] == "helmet"]
        vests   = [d for d in detections if d["class_name"] == "safety_vest"]
        fires   = [d for d in detections if d["class_name"] == "fire"]
        smokes  = [d for d in detections if d["class_name"] == "smoke"]

        # ---------------------------------------------------------------
        # PEMERIKSAAN PPE PER WORKER
        # ---------------------------------------------------------------
        for track in tracks:

            # Hanya periksa kelas person (class_id = 0)
            if track["class_id"] != 0:
                continue

            person_bbox = track["bbox"]

            # Periksa keberadaan helm (area 35% atas = area kepala)
            has_helmet = any(
                self._is_center_inside_region(
                    h["bbox"], person_bbox,
                    y_min_ratio=0.0,
                    y_max_ratio=0.35
                )
                for h in helmets
            )

            # Periksa keberadaan rompi (area 25%-100% = torso ke bawah)
            has_vest = any(
                self._is_center_inside_region(
                    v["bbox"], person_bbox,
                    y_min_ratio=0.25,
                    y_max_ratio=1.0
                )
                for v in vests
            )

            # Tentukan status berdasarkan keberadaan PPE
            if not has_helmet:
                status = "NO_HELMET"
            elif not has_vest:
                status = "NO_SAFETY_VEST"
            else:
                status = "SAFE"

            violations.append({
                "track_id": track["track_id"],
                "status":   status
            })

        # ---------------------------------------------------------------
        # DETEKSI BAHAYA KEBAKARAN / ASAP
        # ---------------------------------------------------------------
        if fires:
            violations.append({"status": "FIRE_ALERT"})

        if smokes:
            violations.append({"status": "SMOKE_ALERT"})

        return violations