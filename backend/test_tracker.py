from backend.detector import PlantationDetector
from backend.tracker import PlantationTracker

from pathlib import Path
import cv2

# =====================
# Path test image
# =====================
BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "test.jpg"

# =====================
# Load image
# =====================
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    print("Gagal membaca gambar")
    exit()

# =====================
# Initialize
# =====================
detector = PlantationDetector()

tracker = PlantationTracker()

# =====================
# Detection
# =====================
detections = detector.detect(image)

print("\nDetections:")
print(detections)

# =====================
# Tracking
# =====================
tracks = tracker.update(detections)

print("\nTracks:")
print(tracks)