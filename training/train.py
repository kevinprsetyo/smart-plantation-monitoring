from ultralytics import YOLO
from pathlib import Path

# =====================================
# Path Project
# =====================================
BASE_DIR = Path(r"D:\Portofolio Vercel\smart-plantation-monitoring")

# Dataset
DATA_YAML = BASE_DIR / "datasets" / "merged" / "data.yaml"

# Output model
PROJECT_DIR = BASE_DIR / "models"
RUN_NAME = "smart_plantation_yolov8s"

# Checkpoint terakhir
LAST_CHECKPOINT = (
    PROJECT_DIR
    / RUN_NAME
    / "weights"
    / "last.pt"
)


def main():

    # =====================================
    # Resume Training
    # =====================================
    if LAST_CHECKPOINT.exists():

        print("\n===================================")
        print(" Resume training from last.pt")
        print("===================================\n")

        model = YOLO(str(LAST_CHECKPOINT))

        model.train(
            resume=True
        )

    # =====================================
    # Start Training From Scratch
    # =====================================
    else:

        print("\n===================================")
        print(" Start training from scratch")
        print("===================================\n")

        model = YOLO("yolov8s.pt")

        model.train(

            # Dataset
            data=str(DATA_YAML),

            # =====================================
            # Training
            # =====================================
            epochs=100,
            patience=30,

            # =====================================
            # Image
            # =====================================
            imgsz=640,
            batch=16,

            # =====================================
            # Hardware
            # =====================================
            device=0,
            workers=8,

            # =====================================
            # Optimizer
            # =====================================
            optimizer="AdamW",
            cos_lr=True,

            # =====================================
            # Mixed Precision
            # =====================================
            amp=True,

            # =====================================
            # Dataset Cache
            # =====================================
            cache=True,

            # =====================================
            # Reproducibility
            # =====================================
            seed=42,
            deterministic=True,

            # =====================================
            # Save Model
            # =====================================
            save=True,
            save_period=1,

            # =====================================
            # Output Directory
            # =====================================
            project=str(PROJECT_DIR),
            name=RUN_NAME,

            # =====================================
            # Close Mosaic at Last Epochs
            # =====================================
            close_mosaic=10,

            # =====================================
            # Augmentation
            # =====================================
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,

            degrees=10,
            translate=0.1,
            scale=0.5,
            shear=2.0,

            fliplr=0.5,
            flipud=0.0,

            mosaic=1.0,
            mixup=0.1,

            # =====================================
            # Validation
            # =====================================
            val=True,

            # =====================================
            # Visualization
            # =====================================
            plots=True
        )


if __name__ == "__main__":
    main()