"""Generate synthetic Sudoku digit data, train the digit CNN, and export TFLite.

Run with no arguments:  python development/train_vision.py

Pipeline per synthetic sample:
  1. Render a raw grayscale cell crop (125x125 = 100px cell + 12px margin,
     matching extract_cell() with BOARD_IMAGE_SIZE=900) with one digit,
     grid-line fragments, lighting, blur, noise and mild residual warp.
  2. Run the REAL preprocess_digit() from src/vision/vis_cells.py on that crop.
  3. Keep the resulting 28x28 canvas (or discard it if the real gate rejects it,
     counting rejections so problem digits are visible).
Validation uses fonts that never appear in training.

Then trains the CNN, reports accuracy on held-out fonts and (if present) on real
cells in real_cells/<digit>/*.png, and writes assets/vis_model.keras
and assets/vis_model.tflite (previous copies are kept as *_prev).
Predicted digit = argmax + 1, so inference in vis_cells.py is unchanged.
"""

from types import SimpleNamespace
import importlib.util
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- Geometry (must match the OCR module: BOARD_IMAGE_SIZE=900, margin 0.25) ---
CELL = 100
CROP = 125
MARGIN = (CROP - CELL) // 2  # 12
S = 2                        # supersampling factor for rendering
CHUNK = 500                  # samples per worker task
FONT_EXTS = {".ttf", ".otf", ".ttc"}

# --- Settings (edit here; the script takes no launch arguments) ---
ROOT = Path(__file__).resolve().parents[1]          # .../sudoku
FONT_DIR = ROOT / "assets" / "fonts"
OCR_PATH = ROOT / "src" / "vision" / "vis_cells.py"  # defines preprocess_digit
REAL_DIR = ROOT / "real_cells"                       # <digit>/*.png real test cells
DATA_FILE = ROOT / "development" / "sudoku_digits.npz"
KERAS_OUT = ROOT / "assets" / "vis_model.keras"
TFLITE_OUT = ROOT / "assets" / "vis_model.tflite"
REGENERATE_DATA = True      # False = reuse DATA_FILE if it exists
TRAIN_PER_CLASS = 5000 #20000
VAL_PER_CLASS = 2000
VAL_FONT_FRAC = 0.15
EPOCHS = 15 #40
BATCH_SIZE = 256            # larger batches suit the M1 GPU; small CNN, so no accuracy cost here
USE_GPU = True              # Metal GPU if tensorflow-metal is installed; set False to compare against CPU
SEED = 0
WORKERS = max(1, (os.cpu_count() or 2) - 1)   # leave one core for the OS


# --------------------------------------------------------------------------
# Fonts / glyphs
# --------------------------------------------------------------------------

def render_glyph(font, ch):
    """Render one character and return its tight-cropped uint8 mask."""
    img = Image.new("L", (500, 500), 0)
    ImageDraw.Draw(img).text((100, 60), ch, font=font, fill=255)
    bbox = img.getbbox()
    if bbox is None:
        return None
    return np.array(img.crop(bbox))


def find_font_files(font_dir):
    paths = sorted(p for p in Path(font_dir).rglob("*") if p.suffix.lower() in FONT_EXTS)
    if not paths:
        try:
            import matplotlib
            fallback = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
            if fallback.exists():
                print(f"[WARN] No fonts in '{font_dir}'; using fallback {fallback.name}")
                return [fallback]
        except Exception:
            pass
        raise SystemExit(f"No fonts found in '{font_dir}'")
    return paths


def load_glyph_sets(font_dir):
    """Return a list of {digit: mask} dicts, one per usable font."""
    glyph_sets = []
    for path in find_font_files(font_dir):
        try:
            font = ImageFont.truetype(str(path), 160)
            masks = {d: render_glyph(font, str(d)) for d in range(1, 10)}
        except Exception as exc:
            print(f"[SKIP] {path.name}: {exc}")
            continue
        if any(m is None or min(m.shape) < 10 for m in masks.values()):
            print(f"[SKIP] {path.name}: missing/tiny glyphs")
            continue
        keys = {cv2.resize(m, (24, 24), interpolation=cv2.INTER_AREA).tobytes() for m in masks.values()}
        if len(keys) != 9:
            print(f"[SKIP] {path.name}: duplicate glyphs (placeholder boxes?)")
            continue
        heights = [m.shape[0] for m in masks.values()]
        if max(heights) / min(heights) > 1.3:
            print(f"[SKIP] {path.name}: uneven digit heights (old-style numerals?)")
            continue
        glyph_sets.append(masks)
    if not glyph_sets:
        raise SystemExit("No usable fonts")
    return glyph_sets


# --------------------------------------------------------------------------
# Crop synthesis
# --------------------------------------------------------------------------

def paste_max(canvas, patch, x0, y0):
    h, w = patch.shape
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(canvas.shape[1], x0 + w), min(canvas.shape[0], y0 + h)
    if xb <= xa or yb <= ya:
        return
    sub = patch[ya - y0:yb - y0, xa - x0:xb - x0]
    canvas[ya:yb, xa:xb] = np.maximum(canvas[ya:yb, xa:xb], sub)


def draw_digit(mask, glyph, rng):
    W = mask.shape[0]
    target_h = rng.uniform(0.42, 0.72) * CELL * S
    h0, w0 = glyph.shape
    new_h = max(4, int(target_h))
    new_w = max(2, int(w0 * (target_h / h0) * rng.uniform(0.85, 1.15)))
    if new_w > 0.8 * CELL * S:
        new_w = int(0.8 * CELL * S)
    g = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    # Ink thickness variation (bold / light)
    t = int(rng.choice([-1, 0, 0, 1, 1, 2, 3]))
    if t > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))
        g = cv2.dilate(g, k)
    elif t < 0:
        g = cv2.erode(g, np.ones((3, 3), np.uint8))

    cx = W / 2 + rng.uniform(-0.06, 0.06) * CELL * S
    cy = W / 2 + rng.uniform(-0.06, 0.06) * CELL * S
    paste_max(mask, g, int(cx - g.shape[1] / 2), int(cy - g.shape[0] / 2))


def draw_grid_lines(mask, rng):
    W = mask.shape[0]
    lo = MARGIN * S
    hi = (MARGIN + CELL) * S
    for side in range(4):
        if rng.random() > 0.92:
            continue
        pos = (lo if side in (0, 1) else hi) + rng.normal(0, 2 * S)
        thick = int(rng.choice([1, 2, 3, 3, 4, 6]) * S)
        tilt = rng.uniform(-2, 2) * S
        intensity = rng.uniform(0.6, 1.0)
        if side in (0, 2):   # vertical
            p1, p2 = (int(pos - tilt), 0), (int(pos + tilt), W)
        else:                # horizontal
            p1, p2 = (0, int(pos - tilt)), (W, int(pos + tilt))
        line = np.zeros_like(mask)
        cv2.line(line, p1, p2, float(intensity), thick)
        np.maximum(mask, line, out=mask)


def random_homography(rng, W):
    c = W / 2
    ang = np.deg2rad(rng.uniform(-3, 3))
    shear = rng.uniform(-0.08, 0.08)
    sc = rng.uniform(0.95, 1.05)
    A = np.array([[sc * np.cos(ang), -sc * np.sin(ang) + shear, 0],
                  [sc * np.sin(ang), sc * np.cos(ang), 0],
                  [0, 0, 1]], dtype=np.float64)
    T1 = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]], dtype=np.float64)
    T2 = np.array([[1, 0, c], [0, 1, c], [0, 0, 1]], dtype=np.float64)
    src = np.float32([[0, 0], [W, 0], [W, W], [0, W]])
    dst = (src + rng.normal(0, 0.012 * W, (4, 2))).astype(np.float32)
    P = cv2.getPerspectiveTransform(src, dst)
    return P @ T2 @ A @ T1


def add_paper_noise(img, rng):
    """Chunky, camera/paper-like noise instead of per-pixel speckle.

    - very light fine grain
    - blotchy low-frequency texture (noise generated at low resolution, then upsampled)
    - discrete dark flecks, which become the white specks seen after inverse thresholding
    """
    h, w = img.shape
    img = img + rng.normal(0, rng.uniform(0, 2.5), img.shape)

    if rng.random() < 0.7:
        k = int(rng.choice([3, 4, 6, 8]))
        low = rng.normal(0, rng.uniform(3, 12), (h // k + 2, w // k + 2)).astype(np.float32)
        img = img + cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)

    if rng.random() < 0.8:
        layer = np.zeros((h, w), np.float32)
        for _ in range(int(rng.integers(0, 25))):
            center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            radius = max(1, int(round(rng.uniform(0.8, 3.5))))
            cv2.circle(layer, center, radius, float(rng.uniform(25, 110)), -1)
        img = img - cv2.GaussianBlur(layer, (0, 0), 0.7)
    return img


_GRID = {}


def _grid(W):
    if W not in _GRID:
        _GRID[W] = np.mgrid[0:W, 0:W].astype(np.float32) / W - 0.5
    return _GRID[W]


def make_crop(glyph, rng):
    W = CROP * S
    mask = np.zeros((W, W), np.float32)
    draw_digit(mask, glyph, rng)
    draw_grid_lines(mask, rng)

    H = random_homography(rng, W)
    mask = cv2.warpPerspective(mask, H, (W, W), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

    paper = rng.uniform(150, 255)
    ink = rng.uniform(0, 90)
    if paper - ink < 70:
        ink = max(0.0, paper - 70)
    img = paper - (paper - ink) * np.clip(mask, 0, 1)
    # Degradation-free reference of the same crop (used to reject samples the noise destroyed)
    clean = cv2.resize(img, (CROP, CROP), interpolation=cv2.INTER_AREA).astype(np.uint8)

    # Lighting gradient
    yy, xx = _grid(W)
    theta = rng.uniform(0, 2 * np.pi)
    grad = 1 + rng.uniform(-0.25, 0.25) * 2 * (xx * np.cos(theta) + yy * np.sin(theta))
    img = np.clip(img * grad, 0, 255)

    # Focus blur, then downsample to the real crop size
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.6) * S)
    img = cv2.resize(img, (CROP, CROP), interpolation=cv2.INTER_AREA)

    # Sometimes emulate a lower-resolution source (down then up)
    if rng.random() < 0.3:
        f = rng.uniform(0.5, 0.9)
        small = cv2.resize(img, (int(CROP * f), int(CROP * f)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (CROP, CROP), interpolation=cv2.INTER_LINEAR)

    img = add_paper_noise(img, rng)
    img = np.clip(img, 0, 255).astype(np.uint8)

    if rng.random() < 0.5:
        q = int(rng.integers(35, 96))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return img, clean


# --------------------------------------------------------------------------
# Real preprocessing + parallel workers
# --------------------------------------------------------------------------

def load_preprocess(ocr_path):
    ocr_path = Path(ocr_path).resolve()
    spec = importlib.util.spec_from_file_location("ocr_module", ocr_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ocr_module"] = module
    spec.loader.exec_module(module)
    return module.preprocess_digit


_STATE = {}


def _init_worker(glyphs, ocr_path):
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    _STATE["glyphs"] = glyphs
    _STATE["pre"] = load_preprocess(ocr_path)


MIN_CANVAS_IOU = 0.5    # reject samples whose processed canvas no longer resembles the clean digit (safety net)


def canvases_agree(canvas, reference):
    """IoU of the (slightly dilated) processed canvas vs the same crop processed without degradation."""
    k = np.ones((3, 3), np.uint8)
    a = cv2.dilate((canvas > 0).astype(np.uint8), k)
    b = cv2.dilate((reference > 0).astype(np.uint8), k)
    union = np.logical_or(a, b).sum()
    return union > 0 and np.logical_and(a, b).sum() / union >= MIN_CANVAS_IOU


def _work(task):
    split, digit, n, seed_seq = task
    rng = np.random.default_rng(seed_seq)
    glyph_sets = _STATE["glyphs"][split]
    pre = _STATE["pre"]
    out, tried = [], 0
    while len(out) < n and tried < n * 20:
        tried += 1
        glyph = glyph_sets[int(rng.integers(len(glyph_sets)))][digit]
        crop, clean = make_crop(glyph, rng)
        canvas = pre(crop)
        if canvas is None:
            continue
        reference = pre(clean)
        if reference is None or not canvases_agree(canvas, reference):
            continue
        out.append(canvas)
    arr = np.stack(out) if out else np.zeros((0, 28, 28), np.uint8)
    return split, digit, arr, tried


def save_preview(x, y, path, per_digit=16):
    rows = []
    for d in range(1, 10):
        idx = np.where(y == d)[0][:per_digit]
        row = np.concatenate([x[i] for i in idx], axis=1) if len(idx) else np.zeros((28, 28), np.uint8)
        pad = per_digit * 28 - row.shape[1]
        rows.append(np.pad(row, ((0, 0), (0, pad))))
    cv2.imwrite(str(path), np.concatenate(rows, axis=0))


def generate_data():
    args = SimpleNamespace(
        ocr_path=OCR_PATH,
        font_dir=FONT_DIR,
        train_per_class=TRAIN_PER_CLASS,
        val_per_class=VAL_PER_CLASS,
        val_font_frac=VAL_FONT_FRAC,
        out=DATA_FILE,
        seed=SEED,
        workers=WORKERS,
    )
    print(f"[INFO] Using OCR module: {args.ocr_path}")

    glyph_sets = load_glyph_sets(args.font_dir)
    print(f"[INFO] {len(glyph_sets)} usable fonts")

    order = np.random.default_rng(args.seed).permutation(len(glyph_sets))
    n_val_fonts = int(round(args.val_font_frac * len(glyph_sets)))
    if len(glyph_sets) < 4 or n_val_fonts < 1:
        print("[WARN] Too few fonts to hold any out: validation shares fonts with training")
        glyphs = {"train": glyph_sets, "val": glyph_sets}
    else:
        val_idx = set(order[:n_val_fonts].tolist())
        glyphs = {
            "val": [glyph_sets[i] for i in sorted(val_idx)],
            "train": [glyph_sets[i] for i in range(len(glyph_sets)) if i not in val_idx],
        }
        print(f"[INFO] {len(glyphs['train'])} train fonts, {len(glyphs['val'])} held-out val fonts")

    per_class = {"train": args.train_per_class, "val": args.val_per_class}
    tasks = []
    for split, total in per_class.items():
        for digit in range(1, 10):
            remaining = total
            while remaining > 0:
                n = min(CHUNK, remaining)
                tasks.append((split, digit, n, None))
                remaining -= n
    seeds = np.random.SeedSequence(args.seed).spawn(len(tasks))
    tasks = [(s, d, n, ss) for (s, d, n, _), ss in zip(tasks, seeds)]

    results = {(s, d): [] for s in per_class for d in range(1, 10)}
    tried = {(s, d): 0 for s in per_class for d in range(1, 10)}
    with Pool(args.workers, initializer=_init_worker, initargs=(glyphs, args.ocr_path)) as pool:
        for i, (split, digit, arr, t) in enumerate(pool.imap_unordered(_work, tasks), 1):
            results[(split, digit)].append(arr)
            tried[(split, digit)] += t
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} chunks", flush=True)

    out = {}
    print("\nAcceptance rate through your preprocess_digit gate + digit-integrity check (low = problem digit):")
    for split in per_class:
        xs, ys = [], []
        for digit in range(1, 10):
            arr = np.concatenate(results[(split, digit)])[:per_class[split]]
            rate = sum(len(a) for a in results[(split, digit)]) / max(1, tried[(split, digit)])
            print(f"  {split:5s} digit {digit}: {rate:6.1%}  ({len(arr)} kept)")
            xs.append(arr)
            ys.append(np.full(len(arr), digit, np.uint8))
        x, y = np.concatenate(xs), np.concatenate(ys)
        perm = np.random.default_rng(args.seed + 1).permutation(len(x))
        out[f"x_{split}"], out[f"y_{split}"] = x[perm], y[perm]

    np.savez_compressed(args.out, **out)
    preview_path = Path(args.out).with_name("preview_train.png")
    save_preview(out["x_train"], out["y_train"], preview_path)
    print(f"\nSaved {args.out}  (train {out['x_train'].shape}, val {out['x_val'].shape})")
    print(f"Saved {preview_path}: eyeball it against your real preprocess_digit canvases")



# --------------------------------------------------------------------------
# Training (TensorFlow is imported lazily so worker processes stay light)
# --------------------------------------------------------------------------

def load_real(real_dir):
    xs, ys, names = [], [], []
    for d in range(1, 10):
        for p in sorted((Path(real_dir) / str(d)).glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (28, 28):
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            xs.append(img)
            ys.append(d)
            names.append(str(p))
    if not xs:
        return None
    return np.array(xs, np.uint8), np.array(ys), names


def prep(x):
    return x.reshape(-1, 28, 28, 1).astype("float32") / 255.0


def build_model():
    from tensorflow.keras import layers, models
    return models.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Dropout(0.5),
        layers.Dense(9, activation="softmax"),
    ])


def report(model, x, y, title, names=None):
    from sklearn.metrics import confusion_matrix
    probs = model.predict(prep(x), batch_size=1024, verbose=0)
    pred = probs.argmax(1) + 1
    print(f"\n=== {title}: accuracy {np.mean(pred == y):.4f} ({len(y)} samples) ===")
    print("Confusion matrix (rows = true 1-9, cols = predicted 1-9):")
    print(confusion_matrix(y, pred, labels=list(range(1, 10))))
    for d in range(1, 10):
        m = y == d
        if m.any():
            print(f"  recall {d}: {np.mean(pred[m] == d):.3f}")
    if names is not None:
        wrong = [(n, t, p) for n, t, p in zip(names, y, pred) if t != p]
        if wrong:
            print("Misclassified real cells (file, true, predicted):")
            for n, t, p in wrong[:40]:
                print(f"  {n}  {t} -> {p}")


def backup(path):
    path = Path(path)
    if path.exists():
        shutil.copy2(path, path.with_name(f"{path.stem}_prev{path.suffix}"))


def train():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    if not USE_GPU:
        tf.config.set_visible_devices([], "GPU")
    print("[INFO] TensorFlow devices:", [d.name for d in tf.config.list_physical_devices()])

    data = np.load(DATA_FILE)

    def make_ds(x_u8, y, shuffle):
        # Data stays uint8 in memory; scaling to [0,1] and one-hot (needed for label
        # smoothing) happen per batch. Normalisation stays OUTSIDE the model so
        # inference in vis_cells.py (which divides by 255) is unchanged.
        ds = tf.data.Dataset.from_tensor_slices((x_u8, y.astype(np.int32) - 1))
        if shuffle:
            ds = ds.shuffle(len(y), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.batch(BATCH_SIZE).map(
            lambda x, lbl: (tf.cast(tf.reshape(x, (-1, 28, 28, 1)), tf.float32) / 255.0, tf.one_hot(lbl, 9)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(data["x_train"], data["y_train"], shuffle=True)
    val_ds = make_ds(data["x_val"], data["y_val"], shuffle=False)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    real = load_real(REAL_DIR) if REAL_DIR.exists() else None
    if real is not None:
        print(f"[INFO] Loaded {len(real[1])} real cells from {REAL_DIR}")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    if real is not None:
        class RealEval(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                pred = self.model.predict(prep(real[0]), batch_size=1024, verbose=0).argmax(1) + 1
                print(f"    real-cell accuracy: {np.mean(pred == real[1]):.3f} ({len(real[1])} cells)")
        callbacks.append(RealEval())

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    report(model, data["x_val"], data["y_val"].astype(int), "Synthetic validation (held-out fonts)")

    if real is not None:
        report(model, real[0], real[1], "REAL cells (the number that matters)", real[2])
    else:
        print(f"\n[INFO] No real test set at '{REAL_DIR}'. Synthetic accuracy alone is not trustworthy.")

    backup(KERAS_OUT)
    backup(TFLITE_OUT)
    model.save(KERAS_OUT)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    TFLITE_OUT.write_bytes(converter.convert())
    print(f"\nSaved {KERAS_OUT} and {TFLITE_OUT}")


def main():
    if REGENERATE_DATA or not DATA_FILE.exists():
        generate_data()
    train()


if __name__ == "__main__":
    main()