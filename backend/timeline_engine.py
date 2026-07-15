"""
timeline_engine.py
==================
Modul pencatatan event pelanggaran berbasis waktu selama pemrosesan video.

TimelineEngine mencatat setiap perubahan status pelanggaran per track_id
selama pemrosesan video berlangsung. Dengan hanya mencatat perubahan status
(bukan setiap frame), timeline yang dihasilkan informatif dan bebas duplikat.

Contoh output event:
    [
        {"time": "00:00:05", "track_id": 3, "status": "NO_HELMET"},
        {"time": "00:00:12", "track_id": 7, "status": "NO_SAFETY_VEST"},
        {"time": "00:00:18", "track_id": 3, "status": "SAFE"}
    ]

Integrasi:
    - Diinisialisasi di VideoProcessor.process_video() setelah fps diketahui.
    - Dipanggil per frame setelah ViolationEngine.check().
    - Hasilnya disertakan di return value process_video() sebagai "timeline_events".
"""


class TimelineEngine:
    """
    Pencatat event pelanggaran berbasis timestamp video.

    Melacak status terakhir setiap track_id dan mencatat event baru
    hanya ketika terjadi perubahan status, sehingga timeline tidak
    dibanjiri baris yang sama berulang-ulang.

    Attributes:
        fps (float)         : Frame rate video, digunakan untuk konversi
                              nomor frame ke timestamp HH:MM:SS.
        events (list)       : Daftar event yang telah dicatat.
        _last_status (dict) : Cache status terakhir per track_id
                              untuk mendeteksi perubahan.
    """

    def __init__(self, fps: float):
        """
        Inisialisasi timeline engine untuk satu sesi pemrosesan video.

        Args:
            fps (float): Frame rate video input. Jika <= 0 atau tidak valid,
                         akan diganti dengan nilai default 30.0.
        """
        self.fps          = fps if fps > 0 else 30.0
        self.events       = []
        self._last_status = {}   # {track_id (int): status (str)}

    # ------------------------------------------------------------------
    # PRIVATE METHODS
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_to_timestamp(frame_count: int, fps: float) -> str:
        """
        Konversi nomor frame ke format timestamp HH:MM:SS.

        Args:
            frame_count (int): Nomor frame saat ini (1-indexed).
            fps (float)       : Frame rate video.

        Returns:
            str: Timestamp dalam format "HH:MM:SS".
                 Contoh: frame 150 pada 30fps -> "00:00:05".
        """
        total_seconds = int(frame_count / fps)
        hours   = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------

    def record(self, frame_count: int, violations: list) -> None:
        """
        Catat event pelanggaran untuk frame saat ini.

        Hanya event dengan perubahan status yang dicatat. Jika status
        seorang worker tidak berubah dari frame sebelumnya, tidak ada
        entry baru yang ditambahkan ke self.events.

        Event untuk FIRE_ALERT dan SMOKE_ALERT (tanpa track_id)
        tidak dicatat karena tidak memiliki identitas track.

        Args:
            frame_count (int): Nomor frame saat ini (untuk kalkulasi timestamp).
            violations (list[dict]): Output dari ViolationEngine.check().
                Setiap item yang memiliki "track_id" akan dievaluasi.
        """
        timestamp = self._frame_to_timestamp(frame_count, self.fps)

        for v in violations:

            # Hanya proses event yang memiliki track_id (worker violations)
            if "track_id" not in v:
                continue

            track_id = v["track_id"]
            status   = v["status"]

            # Catat hanya jika status berubah dari event terakhir worker ini
            if self._last_status.get(track_id) != status:
                self._last_status[track_id] = status
                self.events.append({
                    "time":     timestamp,
                    "track_id": track_id,
                    "status":   status
                })

    def get_events(self) -> list:
        """
        Kembalikan seluruh event yang telah dicatat.

        Returns:
            list[dict]: Daftar event timeline, diurutkan berdasarkan urutan
                kejadian (kronologis). Setiap item berisi:
                {
                    "time"     : str,  # timestamp HH:MM:SS
                    "track_id" : int,  # ID worker
                    "status"   : str   # status pelanggaran
                }
                Mengembalikan list kosong jika tidak ada event yang dicatat.
        """
        return self.events
