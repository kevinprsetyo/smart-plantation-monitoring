"""
detector.py
===========
Modul deteksi objek menggunakan model YOLOv8 yang sudah dilatih khusus
untuk dataset keselamatan perkebunan.

Kelas yang terdeteksi (sesuai urutan class_id model):
    0 : person
    1 : helmet
    2 : safety_vest
    3 : truck
    4 : forklift
    5 : excavator
    6 : fire
    7 : smoke
"""

from ultralytics import YOLO


class PlantationDetector:
    """
    Wrapper model YOLOv8 untuk deteksi objek pada frame gambar/video.

    Attributes:
        model (YOLO): Instance model YOLOv8 yang sudah dimuat dari file .pt.
    """

    def __init__(self):
        """
        Muat model YOLOv8 dari file checkpoint yang sudah dilatih.

        Model dimuat dari backend/models/best.pt.
        """
        self.model = YOLO("backend/models/best.pt")

    def detect(self, frame) -> list:
        """
        Jalankan deteksi objek pada satu frame gambar.

        Menggunakan confidence threshold 0.5 untuk menyaring deteksi
        dengan keyakinan rendah.

        Args:
            frame (numpy.ndarray): Frame gambar dalam format BGR (output OpenCV).

        Returns:
            list[dict]: Daftar deteksi, masing-masing berisi:
                {
                    "class_id"   : int,   # indeks kelas objek
                    "class_name" : str,   # nama kelas objek
                    "confidence" : float, # skor kepercayaan (0.0 - 1.0)
                    "bbox"       : list   # [x1, y1, x2, y2] koordinat piksel
                }
            Mengembalikan list kosong jika tidak ada objek terdeteksi.
        """
        results = self.model.predict(
            source=frame,
            conf=0.5,
            verbose=False
        )

        detections = []
        result = results[0]

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            detections.append({
                "class_id":   class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox":       [x1, y1, x2, y2]
            })

        return detections