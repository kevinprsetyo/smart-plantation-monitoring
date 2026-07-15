"""
video_processor.py
==================
Modul pemrosesan video untuk Smart Plantation Safety Monitoring System.

Alur kerja:
    1. Baca video input frame per frame menggunakan OpenCV.
    2. Jalankan deteksi objek (YOLOv8) pada setiap frame.
    3. Lacak objek yang relevan menggunakan ByteTrack.
    4. Periksa pelanggaran PPE (helm, rompi) dan deteksi kebakaran/asap.
    5. Catat event pelanggaran ke TimelineEngine (hanya saat status berubah).
    6. Tulis frame yang sudah dianotasi ke file sementara (codec mp4v).
    7. Re-encode file sementara ke H.264 menggunakan ffmpeg agar kompatibel
       dengan browser, Streamlit, dan VLC.
    8. Kembalikan statistik, status pelanggaran, jumlah objek, dan timeline.

Catatan codec:
    OpenCV VideoWriter di Windows tidak mendukung H.264 secara langsung.
    Oleh karena itu, output sementara ditulis dengan codec mp4v, kemudian
    di-re-encode oleh ffmpeg dengan flag -movflags +faststart agar file
    dapat di-stream tanpa harus mendownload seluruhnya terlebih dahulu.
"""

import cv2
import subprocess
import os
from pathlib import Path

from backend.detector import PlantationDetector
from backend.tracker import PlantationTracker
from backend.violation_engine import ViolationEngine
from backend.statistics_engine import StatisticsEngine
from backend.snapshot_manager import SnapshotManager
from backend.timeline_engine import TimelineEngine


class VideoProcessor:
    """
    Kelas utama untuk memproses video surveillance.

    Mengkoordinasikan pipeline deteksi -> tracking -> violation check
    -> timeline recording -> anotasi frame -> penulisan video output.

    Attributes:
        detector (PlantationDetector)  : Model YOLOv8 untuk deteksi objek.
        tracker (PlantationTracker)    : ByteTrack untuk pelacakan objek antar frame.
        violation_engine (ViolationEngine) : Logika pemeriksaan pelanggaran PPE.
        snapshot_manager (SnapshotManager) : Penyimpanan snapshot frame pelanggaran.
        statistics_engine (StatisticsEngine): Akumulasi statistik per video.
        timeline_engine (TimelineEngine)   : Pencatatan event pelanggaran per waktu.
        max_object_count (dict)        : Jumlah objek maksimum dalam satu frame.
    """

    def __init__(self):
        """Inisialisasi semua komponen pipeline."""
        self.detector        = PlantationDetector()
        self.tracker         = PlantationTracker()
        self.violation_engine = ViolationEngine()
        self.snapshot_manager = SnapshotManager()

    # ------------------------------------------------------------------
    # PRIVATE METHODS
    # ------------------------------------------------------------------

    def _reencode_to_h264(self, raw_path: str, final_path: str) -> bool:
        """
        Re-encode video dari format mp4v (output mentah OpenCV) ke H.264.

        Menggunakan ffmpeg dengan opsi berikut:
            -c:v libx264       : Codec video H.264
            -preset fast       : Keseimbangan antara kecepatan dan ukuran file
            -crf 23            : Kualitas konstan (skala 0-51, lebih rendah = lebih baik)
            -pix_fmt yuv420p   : Format piksel yang kompatibel dengan semua player
            -movflags +faststart: Memindahkan metadata ke awal file untuk streaming
            -an                : Tidak menyertakan audio (tidak diperlukan untuk CCTV)

        Args:
            raw_path (str)  : Path file mp4v mentah hasil OpenCV VideoWriter.
            final_path (str): Path file output H.264 yang akan dihasilkan.

        Returns:
            bool: True jika re-encode berhasil, False jika gagal.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-i", raw_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            final_path
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode("utf-8", errors="replace")
                print(f"[VideoProcessor] ffmpeg error: {error_msg}")
                return False

            return True

        except FileNotFoundError:
            print("[VideoProcessor] ffmpeg tidak ditemukan. Pastikan ffmpeg ada di PATH sistem.")
            return False

        except subprocess.TimeoutExpired:
            print("[VideoProcessor] ffmpeg melebihi batas waktu (timeout 600 detik).")
            return False

    def _get_violation_status(self, track_id: int, violations: list) -> str:
        """
        Cari status pelanggaran untuk track ID tertentu.

        Args:
            track_id (int): ID unik objek yang sedang dilacak.
            violations (list): Daftar dict pelanggaran dari ViolationEngine.

        Returns:
            str: Status pelanggaran ('SAFE', 'NO_HELMET', 'NO_SAFETY_VEST').
        """
        for v in violations:
            if "track_id" in v and v["track_id"] == track_id:
                return v["status"]
        return "SAFE"

    def _get_bbox_color(self, status: str) -> tuple:
        """
        Tentukan warna bounding box berdasarkan status pelanggaran.

        Args:
            status (str): Status pelanggaran worker.

        Returns:
            tuple: Warna BGR untuk cv2 (B, G, R).
                - SAFE           : Hijau  (0, 255, 0)
                - NO_HELMET      : Merah  (0, 0, 255)
                - NO_SAFETY_VEST : Oranye (0, 165, 255)
        """
        color_map = {
            "SAFE":            (0, 255, 0),
            "NO_HELMET":       (0, 0, 255),
            "NO_SAFETY_VEST":  (0, 165, 255),
        }
        return color_map.get(status, (0, 255, 0))

    def _count_detections(self, detections: list) -> dict:
        """
        Hitung jumlah deteksi per kelas objek dalam satu frame.

        Args:
            detections (list): Daftar dict deteksi dari PlantationDetector.

        Returns:
            dict: Jumlah deteksi per kelas objek.
        """
        classes = [
            "person", "helmet", "safety_vest",
            "truck", "forklift", "excavator",
            "fire", "smoke"
        ]
        return {
            cls: len([d for d in detections if d["class_name"] == cls])
            for cls in classes
        }

    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------

    def process_video(self, input_path: str, output_path: str) -> dict:
        """
        Proses seluruh video: deteksi, tracking, anotasi, dan simpan output.

        Langkah-langkah:
            1. Buka video input dan baca properti (resolusi, FPS).
            2. Inisialisasi StatisticsEngine dan TimelineEngine.
            3. Inisialisasi VideoWriter dengan codec mp4v ke file sementara.
            4. Loop setiap frame:
               a. Deteksi objek.
               b. Update object count maksimum.
               c. Tracking (ByteTrack).
               d. Cek pelanggaran PPE.
               e. Update statistik akumulatif.
               f. Catat event ke TimelineEngine (jika status berubah).
               g. Simpan snapshot frame pelanggaran.
               h. Gambar bounding box dan label.
               i. Gambar overlay alert kebakaran/asap.
               j. Tulis frame ke file sementara.
            5. Release semua resource (try/finally untuk keamanan).
            6. Validasi file sementara tidak kosong.
            7. Re-encode ke H.264 menggunakan ffmpeg.
            8. Hapus file sementara.
            9. Kembalikan statistik, violations, object_count, dan timeline_events.

        Args:
            input_path (str) : Path absolut atau relatif video input.
            output_path (str): Path tujuan video output H.264.

        Returns:
            dict: Hasil pemrosesan dengan struktur:
                {
                    "statistics": {
                        "total_person": int,
                        "safe_worker": int,
                        "no_helmet": int,
                        "no_safety_vest": int,
                        "fire_alert": int,
                        "smoke_alert": int
                    },
                    "violations": list[dict],       # status akhir per worker
                    "object_count": dict,           # jumlah objek maks per frame
                    "timeline_events": list[dict]   # event pelanggaran kronologis
                }

        Raises:
            Exception: Jika video tidak dapat dibuka, VideoWriter gagal dibuat,
                       file output kosong, atau ffmpeg gagal.
        """
        # Reset semua engine setiap kali dipanggil (satu instansi bisa dipakai ulang)
        self.statistics_engine = StatisticsEngine()
        self.max_object_count = {
            "person": 0,
            "helmet": 0,
            "safety_vest": 0,
            "truck": 0,
            "forklift": 0,
            "excavator": 0,
            "fire": 0,
            "smoke": 0
        }

        # Resolusi ke path absolut untuk menghindari working directory mismatch
        input_path  = str(Path(input_path).resolve())
        output_path = str(Path(output_path).resolve())

        # File sementara mp4v (akan dihapus setelah re-encode berhasil)
        raw_output_path = output_path.replace(".mp4", "_raw.mp4")

        cap = None
        out = None

        try:
            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                raise Exception(f"Video tidak dapat dibuka: {input_path}")

            # Baca properti video dari input
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps          = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Validasi FPS: tangani nilai 0 atau NaN
            if fps <= 0 or fps != fps:
                fps = 30.0
                print("[VideoProcessor] FPS tidak valid, menggunakan default 30.0")
            else:
                fps = round(fps, 3)

            if width <= 0 or height <= 0:
                raise Exception(f"Dimensi frame tidak valid: {width}x{height}")

            print(f"[VideoProcessor] Input  : {width}x{height} @ {fps} fps, {total_frames} frames")
            print(f"[VideoProcessor] Output : {output_path}")

            # Inisialisasi TimelineEngine setelah fps diketahui
            self.timeline_engine = TimelineEngine(fps)

            # Inisialisasi VideoWriter dengan codec mp4v (output sementara)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out    = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                raise Exception(f"VideoWriter gagal dibuat: {raw_output_path}")

            frame_count = 0

            # ----------------------------------------------------------------
            # MAIN LOOP: proses setiap frame
            # ----------------------------------------------------------------
            while True:

                success, frame = cap.read()

                if not success:
                    break

                frame_count += 1

                # -- a. Deteksi objek pada frame saat ini --
                detections = self.detector.detect(frame)

                # -- b. Hitung objek per frame dan update nilai maksimum --
                current_count = self._count_detections(detections)
                for key in current_count:
                    self.max_object_count[key] = max(
                        self.max_object_count[key],
                        current_count[key]
                    )

                # -- c. Tracking: update posisi objek antar frame --
                tracks = self.tracker.update(detections)

                # -- d. Periksa pelanggaran PPE dan bahaya lingkungan --
                violations = self.violation_engine.check(detections, tracks)

                # -- e. Akumulasi statistik keseluruhan video --
                self.statistics_engine.update(violations)

                # -- f. Catat event ke timeline (hanya saat status berubah) --
                self.timeline_engine.record(frame_count, violations)

                # -- g. Simpan snapshot frame yang mengandung pelanggaran --
                # [DINONAKTIFKAN] Snapshot dimatikan sementara untuk menghemat memori disk.
                # Aktifkan kembali dengan menghapus komentar di bawah jika diperlukan.
                # for v in violations:
                #     if v["status"] != "SAFE":
                #         self.snapshot_manager.save(frame, v["status"])

                # -- h. Gambar bounding box dan label pada setiap track --
                for track in tracks:

                    x1, y1, x2, y2 = track["bbox"]
                    track_id   = track["track_id"]
                    class_name = self.detector.model.names[track["class_id"]]
                    status     = self._get_violation_status(track_id, violations)
                    color      = self._get_bbox_color(status)
                    label      = f"{class_name} #{track_id} | {status}"

                    # Klem koordinat agar tidak keluar batas frame
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width  - 1, x2)
                    y2 = min(height - 1, y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Sesuaikan posisi label agar tidak terpotong di tepi atas
                    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                    cv2.putText(
                        frame,
                        label,
                        (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                # -- i. Tampilkan overlay teks untuk alert kebakaran/asap --
                y_text = 40

                for v in violations:

                    if v["status"] == "FIRE_ALERT":
                        cv2.putText(
                            frame,
                            "! FIRE ALERT",
                            (20, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            3
                        )
                        y_text += 50

                    elif v["status"] == "SMOKE_ALERT":
                        cv2.putText(
                            frame,
                            "! SMOKE ALERT",
                            (20, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 255),
                            3
                        )
                        y_text += 50

                # -- j. Tulis frame yang sudah dianotasi ke file sementara --
                out.write(frame)

                if frame_count % 30 == 0:
                    print(f"[VideoProcessor] Progress: {frame_count}/{total_frames} frames")

        finally:
            # Pastikan resource selalu dilepas meskipun terjadi exception
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()

        print(f"[VideoProcessor] Selesai: {frame_count} frames ditulis ke file sementara.")

        # ----------------------------------------------------------------
        # VALIDASI FILE SEMENTARA
        # ----------------------------------------------------------------
        raw_size = os.path.getsize(raw_output_path) if os.path.exists(raw_output_path) else 0

        if raw_size == 0:
            raise Exception(f"File sementara kosong atau tidak ada: {raw_output_path}")

        print(f"[VideoProcessor] File sementara: {raw_size / 1024 / 1024:.2f} MB")

        # ----------------------------------------------------------------
        # RE-ENCODE KE H.264 MENGGUNAKAN FFMPEG
        # ----------------------------------------------------------------
        print("[VideoProcessor] Memulai re-encode ke H.264 (faststart)...")

        encode_success = self._reencode_to_h264(raw_output_path, output_path)

        # Hapus file sementara tanpa mempedulikan hasil re-encode
        try:
            os.remove(raw_output_path)
        except OSError:
            pass

        if not encode_success:
            raise Exception("Re-encode ffmpeg gagal. Pastikan ffmpeg tersedia di PATH sistem.")

        # Validasi file output final tidak kosong
        final_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        if final_size == 0:
            raise Exception(f"File output final kosong: {output_path}")

        print(f"[VideoProcessor] Output H.264: {final_size / 1024 / 1024:.2f} MB -> {output_path}")

        # ----------------------------------------------------------------
        # KEMBALIKAN HASIL STATISTIK DAN TIMELINE
        # ----------------------------------------------------------------
        stats         = self.statistics_engine.get_statistics()
        worker_status = self.statistics_engine.get_worker_status()
        timeline      = self.timeline_engine.get_events()

        return {
            "statistics":      stats,
            "violations":      worker_status,
            "object_count":    self.max_object_count,
            "timeline_events": timeline
        }