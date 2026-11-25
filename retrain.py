"""
Retrain brain tumor CNN with the correct architecture:
  - Output activation: softmax (not relu)
  - Loss: sparse_categorical_crossentropy (integer labels)
  - LR: 0.0001 with Adam
  - Epochs: 15 with early stopping
"""
import os, pathlib, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Suppress verbose TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

TRAIN_DIR = pathlib.Path(r"C:\Users\ponug\Downloads\archive (5)\Training")
TEST_DIR  = pathlib.Path(r"C:\Users\ponug\Downloads\archive (5)\Testing")
IMG_SIZE  = (128, 128)
BATCH     = 32
EPOCHS    = 15
SAVE_PATH = pathlib.Path(__file__).parent / "brain_tumor_cnn.keras"

# ── Data pipeline ────────────────────────────────────────────────────────────
print("Loading datasets …")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH, seed=42
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH, shuffle=False
)

CLASS_NAMES = train_ds.class_names
print("Classes:", CLASS_NAMES)

norm = tf.keras.layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (norm(x), y)).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.map(lambda x, y: (norm(x), y)).cache().prefetch(tf.data.AUTOTUNE)

# ── Model (fixed) ────────────────────────────────────────────────────────────
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    # Data augmentation to improve generalisation
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(CLASS_NAMES), activation="softmax"),   # ← FIXED
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",                  # ← FIXED
    metrics=["accuracy"],
)
model.summary()

# ── Training ─────────────────────────────────────────────────────────────────
cb = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
]

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    callbacks=cb,
)

# ── Evaluation ───────────────────────────────────────────────────────────────
loss, acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest Accuracy: {acc * 100:.2f}%   Test Loss: {loss:.4f}")

# Per-class report
from sklearn.metrics import classification_report
y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ── Save ─────────────────────────────────────────────────────────────────────
model.save(SAVE_PATH)
print(f"Model saved → {SAVE_PATH}")
