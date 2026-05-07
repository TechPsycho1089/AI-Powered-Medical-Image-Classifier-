"""
src/config.py  —  Central configuration for the entire project.
Change values here to tune the model.
"""

CONFIG = {
    # ── Paths ──────────────────────────────────────────────
    "data_dir":       "data/cell_images",
    "infected_dir":   "data/cell_images/Parasitized",
    "healthy_dir":    "data/cell_images/Uninfected",
    "model_save":     "models/malaria_cnn_best.h5",
    "results_dir":    "results",

    # ── Image ──────────────────────────────────────────────
    "img_size":       (64, 64),       # width × height in pixels
    "channels":       3,                # RGB

    # ── Training ───────────────────────────────────────────
    "batch_size":     32,
    "epochs":         30,
    "learning_rate":  1e-4,
    "val_split":      0.15,             # 15% validation
    "test_split":     0.15,             # 15% test
    "threshold":      0.5,              # sigmoid decision threshold

    # ── Labels ─────────────────────────────────────────────
    "classes":        ["Healthy", "Infected"],
    "class_map":      {0: "Healthy", 1: "Infected"},

    # ── Grad-CAM ───────────────────────────────────────────
    "last_conv_layer": "last_conv",     # Must match layer name in model.py
}
