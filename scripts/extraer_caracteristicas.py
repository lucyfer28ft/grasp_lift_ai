import os
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import skew, kurtosis
from scipy.signal import welch

# Rutas
input_dir = r"C:\Users\luciaft\Documents\TFG\TFG\graspAndLiftDetectionTFGProyect\data\processed\ventanas"
output_dir = r"C:\Users\luciaft\Documents\TFG\TFG\graspAndLiftDetectionTFGProyect\data\processed\ventanas\caract"
os.makedirs(output_dir, exist_ok=True)

# Función para cargar partes ordenadas
def load_ordered_parts(prefix):
    part_files = sorted(
        [f for f in os.listdir(input_dir) if f.startswith(prefix) and f.endswith(".npy")],
        key=lambda x: int(x.split("part")[1].split(".")[0])  # orden por índice
    )
    print(f"Cargando partes de {prefix}:", part_files)
    parts = [np.load(os.path.join(input_dir, f)) for f in part_files]
    return np.concatenate(parts, axis=0)

# Cargar y reconstruir X_train_win y X_valid_win
X_train_win = load_ordered_parts("X_train_win_part")
X_valid_win = load_ordered_parts("X_valid_win_part")

# ========================
# FUNCIONES DE CARACTERÍSTICAS
# ========================

# Función para estadísticas temporales
def extract_features_from_window(window):
    stats = []
    for ch in window.T:
        stats.extend([
            np.mean(ch), np.std(ch), np.min(ch), np.max(ch), np.ptp(ch),
            np.median(ch), np.percentile(ch, 25), np.percentile(ch, 75),
            np.mean(np.gradient(ch)), skew(ch), kurtosis(ch)
        ])
    return stats

def extract_time_features_parallel(X_windows, n_jobs=-1):
    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(extract_features_from_window)(win) for win in X_windows
        ),
        dtype=np.float32
    )

# Función para características en frecuencia
def extract_freq_features_from_window(window, fs=128):
    features = []
    for ch in window.T:
        freqs, psd = welch(ch, fs=fs, nperseg=128)
        for low, high in [(0.5, 4), (4, 8), (8, 13), (13, 30)]:
            features.append(np.sum(psd[(freqs >= low) & (freqs < high)]))
        features.append(np.sum(psd))  # energía total
    return features

def extract_freq_features_parallel(X_windows, fs=128, n_jobs=-1):
    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(extract_freq_features_from_window)(win, fs) for win in X_windows
        ),
        dtype=np.float32
    )

# ========================
# EJECUCIÓN
# ========================

print("Extrayendo características TEMPORALES para X_train...")
X_train_feats = extract_time_features_parallel(X_train_win)
np.save(os.path.join(output_dir, "X_train_feats.npy"), X_train_feats)

print("Extrayendo características TEMPORALES para X_valid...")
X_valid_feats = extract_time_features_parallel(X_valid_win)
np.save(os.path.join(output_dir, "X_valid_feats.npy"), X_valid_feats)

print("Extrayendo características en FRECUENCIA para X_train...")
X_train_freq = extract_freq_features_parallel(X_train_win)
np.save(os.path.join(output_dir, "X_train_freq.npy"), X_train_freq)

print("Extrayendo características en FRECUENCIA para X_valid...")
X_valid_freq = extract_freq_features_parallel(X_valid_win)
np.save(os.path.join(output_dir, "X_valid_freq.npy"), X_valid_freq)

print("✅ Extracción completada y archivos guardados.")
