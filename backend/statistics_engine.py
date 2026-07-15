"""
statistics_engine.py
====================
Modul akumulasi statistik pelanggaran PPE selama pemrosesan video.

StatisticsEngine menyimpan status terakhir setiap worker berdasarkan
track_id-nya. Dengan menyimpan status terakhir (bukan menghitung semua
frame), statistik akhir merepresentasikan kondisi keseluruhan setiap
worker yang pernah muncul dalam video.

Contoh penggunaan:
    engine = StatisticsEngine()

    for frame_violations in all_violations:
        engine.update(frame_violations)

    stats = engine.get_statistics()
    worker_list = engine.get_worker_status()
"""


class StatisticsEngine:
    """
    Akumulator statistik PPE untuk pemrosesan video.

    Menyimpan status terakhir setiap worker (berdasarkan track_id) dan
    menghitung total alert kebakaran/asap selama durasi video.

    Attributes:
        worker_status (dict): Peta {track_id: status} status terakhir per worker.
        fire_alert (int)    : Jumlah frame yang mengandung deteksi api.
        smoke_alert (int)   : Jumlah frame yang mengandung deteksi asap.
    """

    def __init__(self):
        """Inisialisasi statistik kosong untuk sesi pemrosesan baru."""
        self.worker_status = {}   # {track_id: status_terakhir}
        self.fire_alert    = 0
        self.smoke_alert   = 0

    def update(self, violations: list) -> None:
        """
        Perbarui statistik berdasarkan hasil violation check satu frame.

        Status worker diperbarui setiap frame sehingga selalu mencerminkan
        status terbaru. Alert kebakaran/asap diakumulasi (dijumlahkan).

        Args:
            violations (list[dict]): Output dari ViolationEngine.check().
                Setiap item adalah dict berisi 'status', dan opsional 'track_id'.
        """
        for v in violations:

            if "track_id" in v:
                # Perbarui status terakhir worker ini
                self.worker_status[v["track_id"]] = v["status"]

            elif v["status"] == "FIRE_ALERT":
                self.fire_alert += 1

            elif v["status"] == "SMOKE_ALERT":
                self.smoke_alert += 1

    def get_statistics(self) -> dict:
        """
        Hitung dan kembalikan ringkasan statistik akhir video.

        Statistik dihitung dari status terakhir setiap worker, sehingga
        satu worker hanya dihitung satu kali meskipun statusnya berubah
        antar frame.

        Returns:
            dict: Statistik dengan struktur:
                {
                    "total_person"  : int,  # total worker unik terdeteksi
                    "safe_worker"   : int,  # worker dengan status terakhir SAFE
                    "no_helmet"     : int,  # worker dengan status terakhir NO_HELMET
                    "no_safety_vest": int,  # worker dengan status terakhir NO_SAFETY_VEST
                    "fire_alert"    : int,  # total frame dengan deteksi api
                    "smoke_alert"   : int   # total frame dengan deteksi asap
                }
        """
        statuses = list(self.worker_status.values())

        return {
            "total_person":   len(statuses),
            "safe_worker":    statuses.count("SAFE"),
            "no_helmet":      statuses.count("NO_HELMET"),
            "no_safety_vest": statuses.count("NO_SAFETY_VEST"),
            "fire_alert":     self.fire_alert,
            "smoke_alert":    self.smoke_alert
        }

    def get_worker_status(self) -> list:
        """
        Kembalikan daftar status terakhir setiap worker yang pernah terdeteksi.

        Returns:
            list[dict]: Daftar status per worker dengan struktur:
                [
                    {"track_id": int, "status": str},
                    ...
                ]
            Mengembalikan list kosong jika tidak ada worker terdeteksi.
        """
        return [
            {"track_id": track_id, "status": status}
            for track_id, status in self.worker_status.items()
        ]