import shutil
from pathlib import Path

BASE_DIR = Path(r"D:\Portofolio Vercel\smart-plantation-monitoring")

DATASETS = {
    "PPE": BASE_DIR / "datasets" / "PPE",
    "Vehicle": BASE_DIR / "datasets" / "Vehicle",
    "Fire": BASE_DIR / "datasets" / "Fire"
}

MERGED_DIR = BASE_DIR / "datasets" / "merged"

# Mapping class lama -> class baru
CLASS_MAP = {
    "PPE": {
        3: 0,  # person
        0: 1,  # helmet
        4: 2   # safety_vest
    },

    "Vehicle": {
        0: 3,  # dump -> truck
        3: 3,  # mixer -> truck
        4: 3,  # moxy -> truck

        2: 4,  # loader -> forklift

        1: 5,  # excavator
        5: 5   # roller -> excavator
    },

    "Fire": {
        0: 6,  # fire
        1: 7   # smoke
    }
}

SPLITS = ["train", "valid", "test"]

for split in SPLITS:

    (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset_path in DATASETS.items():

        img_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"

        if not img_dir.exists():
            continue

        for img_path in img_dir.iterdir():

            new_name = f"{dataset_name}_{img_path.name}"

            shutil.copy(
                img_path,
                MERGED_DIR / split / "images" / new_name
            )

            label_file = label_dir / (img_path.stem + ".txt")

            if not label_file.exists():
                continue

            new_label_path = (
                MERGED_DIR /
                split /
                "labels" /
                (Path(new_name).stem + ".txt")
            )

            new_lines = []

            with open(label_file, "r") as f:

                for line in f.readlines():

                    parts = line.strip().split()

                    old_class = int(parts[0])

                    if old_class not in CLASS_MAP[dataset_name]:
                        continue

                    new_class = CLASS_MAP[dataset_name][old_class]

                    parts[0] = str(new_class)

                    new_lines.append(" ".join(parts))

            with open(new_label_path, "w") as f:
                f.write("\n".join(new_lines))

print("Merge selesai.")