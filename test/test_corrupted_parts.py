import os
import numpy as np

input_dir = r"C:\Users\luciaft\Documents\TFG\TFG\graspAndLiftDetectionTFGProyect\data\processed\ventanas"

def load_ordered_parts(prefix):
    part_files = sorted(
        [f for f in os.listdir(input_dir) if f.startswith(prefix) and f.endswith(".npy")],
        key=lambda x: int(x.split("part")[1].split(".")[0])
    )
    valid_parts = []
    for f in part_files:
        path = os.path.join(input_dir, f)
        arr = np.load(path, mmap_mode="r")  # No carga todo en RAM
        if arr.ndim != 3 or arr.shape[1:] != (128, 32):
            print(f"⚠️ Parte con forma inesperada: {f} -> {arr.shape}")
            continue
        valid_parts.append(np.load(path))
        print(f"✅ {f} cargado: {arr.shape}")
    return np.concatenate(valid_parts, axis=0)

X_train_win = load_ordered_parts("X_train_win_part")
