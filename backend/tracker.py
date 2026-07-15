"""
tracker.py
==========
Modul pelacakan objek menggunakan algoritma ByteTrack via library Supervision.

ByteTrack melacak objek antar frame dengan menetapkan track_id unik yang
konsisten selama objek terlihat dalam video. Pelacakan hanya dilakukan
untuk kelas objek yang relevan: person, truck, forklift, dan excavator.

Referensi:
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box
    https://arxiv.org/abs/2110.06864
"""

import supervision as sv
import numpy as np


class PlantationTracker:
    """
    Wrapper ByteTrack untuk pelacakan objek bergerak dalam video.

    Hanya melacak kelas objek yang memiliki identitas penting:
        - 0 : person    (untuk pemeriksaan PPE)
        - 3 : truck
        - 4 : forklift
        - 5 : excavator

    Kelas seperti helmet, safety_vest, fire, dan smoke tidak dilacak
    karena merupakan atribut/kondisi, bukan entitas bergerak.

    Attributes:
        tracker (sv.ByteTrack): Instance ByteTrack dari library Supervision.
        track_classes (set): Kumpulan class_id yang akan dilacak.
    """

    def __init__(self):
        """Inisialisasi ByteTrack dan tentukan kelas yang dilacak."""
        self.tracker = sv.ByteTrack()

        self.track_classes = {
            0,  # person
            3,  # truck
            4,  # forklift
            5   # excavator
        }

    def update(self, detections: list) -> list:
        """
        Update tracker dengan deteksi dari frame saat ini.

        Menyaring deteksi berdasarkan kelas yang relevan, lalu meneruskan
        ke ByteTrack untuk mendapatkan track_id yang konsisten antar frame.

        Args:
            detections (list[dict]): Output dari PlantationDetector.detect(),
                masing-masing berisi 'class_id', 'bbox', dan 'confidence'.

        Returns:
            list[dict]: Daftar objek yang sedang dilacak, masing-masing berisi:
                {
                    "track_id" : int,  # ID unik dan konsisten antar frame
                    "class_id" : int,  # indeks kelas objek
                    "bbox"     : list  # [x1, y1, x2, y2] koordinat piksel
                }
            Mengembalikan list kosong jika tidak ada deteksi yang relevan.
        """
        # Filter hanya kelas yang perlu dilacak
        track_detections = [
            d for d in detections
            if d["class_id"] in self.track_classes
        ]

        if len(track_detections) == 0:
            return []

        # Konversi ke format yang diharapkan oleh Supervision ByteTrack
        xyxy       = np.array([d["bbox"] for d in track_detections])
        confidence = np.array([d["confidence"] for d in track_detections])
        class_id   = np.array([d["class_id"] for d in track_detections])

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        # Update tracker dan dapatkan hasil dengan track_id
        tracked = self.tracker.update_with_detections(sv_detections)

        results = []

        for i in range(len(tracked.xyxy)):

            x1, y1, x2, y2 = tracked.xyxy[i]

            results.append({
                "track_id": int(tracked.tracker_id[i]),
                "class_id": int(tracked.class_id[i]),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

        return results