"""Train a static SIBI alphabet classifier directly from MediaPipe landmarks.

This follows the MediaPipe landmark + classifier approach of the referenced
sign-language-detector-python project. It reads whole images and detects hands
directly with MediaPipe. The dataset contains 24 static letters; J and Z
require motion and are not included.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATASET = (
    ROOT
    / "data"
    / "sibi"
    / "Labeled Official Indonesian Sign Language (SIBI) Alphabet"
    / "sibi_alphabet_multi_bg_no_augmentation"
)
ALPHABET_CONFIG_PATH = ROOT / "data" / "sibi" / "sibi_alphabet_config.json"
MODEL_PATH = ROOT / "models" / "sibi_landmark_classifier.keras"
LABELS_PATH = ROOT / "models" / "sibi_landmark_labels.json"
REPORT_PATH = ROOT / "models" / "sibi_landmark_training_report.json"
FEATURES_PATH = ROOT / "models" / "sibi_landmark_weights.json"


def letter_from_filename(path: Path) -> str:
    match = re.search(r"(?:^|__)([A-Z])_", path.stem)
    if not match:
        match = re.match(r"([A-Z])_", path.stem)
    if not match:
        raise ValueError(f"Cannot read letter from filename: {path.name}")
    return match.group(1)


def extract_split(hands, split: str, class_to_index: dict[str, int]):
    image_dir = DATASET / "images" / split
    features, labels, skipped = [], [], {"no_hand": 0}
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    for index, image_path in enumerate(image_paths, 1):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((320, 320), Image.Resampling.BILINEAR)
            frame = np.asarray(image, dtype=np.uint8)
        result = hands.process(frame)
        if not result.multi_hand_landmarks:
            skipped["no_hand"] += 1
            continue
        points = result.multi_hand_landmarks[0].landmark
        xs = np.asarray([point.x for point in points], dtype=np.float32)
        ys = np.asarray([point.y for point in points], dtype=np.float32)
        # Match the reference project's translation-invariant 21-point x/y
        # features. The 42 values can also be produced live by MediaPipe Hands.
        feature = np.stack((xs - xs.min(), ys - ys.min()), axis=1).reshape(-1)
        features.append(feature)
        labels.append(class_to_index[letter_from_filename(image_path)])
        if index % 500 == 0:
            print(f"  {split}: processed {index}/{len(image_paths)} images", flush=True)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32), skipped, len(image_paths)


def build_model(class_count: int):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(42,), name="mediapipe_xy_landmarks"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(class_count, activation="softmax", name="letter_probabilities"),
        ],
        name="sibi_static_landmark_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if not DATASET.is_dir():
        raise SystemExit(f"Dataset directory not found: {DATASET}")
    alphabet_config = json.loads(ALPHABET_CONFIG_PATH.read_text(encoding="utf-8"))
    alphabet_order = alphabet_config["alphabet_order"]
    available_letters = {
        letter_from_filename(path)
        for path in (DATASET / "images" / "train").iterdir()
        if path.is_file()
    }
    unexpected_letters = available_letters.difference(alphabet_order)
    if unexpected_letters:
        raise ValueError(f"Dataset has labels outside configured SIBI alphabet: {sorted(unexpected_letters)}")
    letters = [letter for letter in alphabet_order if letter in available_letters]
    class_to_index = {letter: index for index, letter in enumerate(letters)}
    print(f"TensorFlow: {tf.__version__}")
    print(f"MediaPipe: {mp.__version__}")
    print(f"Classes ({len(letters)}): {' '.join(letters)}")
    print("Extracting hand landmarks from full images with MediaPipe only...", flush=True)

    data = {}
    extraction = {}
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.3,
    ) as hands:
        for split in ("train", "val", "test"):
            x, y, skipped, total = extract_split(hands, split, class_to_index)
            data[split] = (x, y)
            extraction[split] = {"detected": len(y), "images": total, "skipped": skipped}
            print(f"{split}: {len(y)}/{total} hands found, skipped={skipped}", flush=True)

    train_x, train_y = data["train"]
    val_x, val_y = data["val"]
    test_x, test_y = data["test"]
    if min(len(train_y), len(val_y), len(test_y)) == 0:
        raise SystemExit("MediaPipe did not find enough hands to train and evaluate.")

    model = build_model(len(letters))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=4, min_lr=1e-5),
    ]
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=120,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )
    test_loss, test_accuracy = model.evaluate(test_x, test_y, verbose=0)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(letters, indent=2), encoding="utf-8")

    # Store inference weights as JSON so the model can later run in the web
    # client without requiring TensorFlow in the Flask server environment.
    weights = [array.tolist() for array in model.get_weights()]
    FEATURES_PATH.write_text(json.dumps({"layers": weights, "classes": letters}, separators=(",", ":")), encoding="utf-8")
    report = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_source": "Teguh Andriyanto, Labeled Official Indonesian Sign Language (SIBI) Alphabet",
        "dataset_license": "CC BY 4.0",
        "alphabet_reference": alphabet_config["reference"],
        "motion_letters": alphabet_config["motion_letters"],
        "method": "MediaPipe Hands on full images; 21 normalized x/y landmarks + Keras MLP",
        "classes": letters,
        "class_count": len(letters),
        "samples": {
            split: len(data[split][1]) for split in ("train", "val", "test")
        },
        "media_pipe_detections": extraction,
        "best_validation_accuracy": float(max(history.history["val_accuracy"])),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "epochs_completed": len(history.history["loss"]),
        "excluded_dynamic_letters": [letter for letter in "JZ" if letter not in letters],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved browser weights: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
