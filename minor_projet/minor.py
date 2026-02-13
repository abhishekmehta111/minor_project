
import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, optimizers, regularizers

# ---------------------- Config ----------------------
DATA_PATH = "/content/final_balanced_medical_history_dataset.csv"
OUT_DIR = "/content/final_hybrid_medical_model"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-4

# Model params
EMBED_DIM = 64
TRANSFORMER_HEADS = 4
TRANSFORMER_FF = 128
LSTM_UNITS = 128
DENSE_UNITS = 256
DROPOUT = 0.30
WEIGHT_DECAY = 1e-5

TEXT_EMBED_DIM = 128
MAX_TOKENS = 20000
TEXT_SEQ_LEN = 20

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# ---------------------- Load Dataset ----------------------
df = pd.read_csv(DATA_PATH).fillna("").drop_duplicates()
print("Dataset shape:", df.shape)

# Identify columns
symptom_cols = [
    c for c in df.columns
    if c not in [
        "diseases","severity",
        "medical_history1","medical_history2","medical_history3",
        "medical_history_effect1","medical_history_effect2","medical_history_effect3"
    ]
]

mh_text_cols = ["medical_history1","medical_history2","medical_history3"]
mh_effect_cols = ["medical_history_effect1","medical_history_effect2","medical_history_effect3"]

# Encode labels
le_d = LabelEncoder()
le_s = LabelEncoder()

df["disease_label"] = le_d.fit_transform(df["diseases"])
df["severity_label"] = le_s.fit_transform(df["severity"])

num_classes_d = len(le_d.classes_)
num_classes_s = len(le_s.classes_)

print("Disease classes:", num_classes_d)
print("Severity classes:", le_s.classes_)

# ---------------------- Prepare Inputs ----------------------
# Symptoms
X_sym = df[symptom_cols].astype("float32").values
seq_len = X_sym.shape[1]
X_sym = X_sym.reshape((-1, seq_len, 1))

# Medical history text
mh1 = df["medical_history1"].str.lower().values
mh2 = df["medical_history2"].str.lower().values
mh3 = df["medical_history3"].str.lower().values

# Effect flags
X_eff = df[mh_effect_cols].astype("float32").values

# Labels
y_d = df["disease_label"].values
y_s = df["severity_label"].values

# ---------------------- Train / Test Split ----------------------
X_sym_tr, X_sym_te, \
mh1_tr, mh1_te, \
mh2_tr, mh2_te, \
mh3_tr, mh3_te, \
X_eff_tr, X_eff_te, \
y_d_tr, y_d_te, \
y_s_tr, y_s_te = train_test_split(
    X_sym, mh1, mh2, mh3, X_eff, y_d, y_s,
    test_size=0.20,
    stratify=y_d,
    random_state=RANDOM_SEED
)

# ---------------------- TextVectorization ----------------------
vect1 = layers.TextVectorization(MAX_TOKENS, output_sequence_length=TEXT_SEQ_LEN)
vect2 = layers.TextVectorization(MAX_TOKENS, output_sequence_length=TEXT_SEQ_LEN)
vect3 = layers.TextVectorization(MAX_TOKENS, output_sequence_length=TEXT_SEQ_LEN)

vect1.adapt(mh1_tr)
vect2.adapt(mh2_tr)
vect3.adapt(mh3_tr)

# ---------------------- Model ----------------------
def build_model():
    # Symptoms
    inp_sym = layers.Input((seq_len,1))
    x = layers.TimeDistributed(layers.Dense(EMBED_DIM))(inp_sym)

    pos = layers.Embedding(seq_len, EMBED_DIM)(tf.range(seq_len))
    x = x + pos

    attn = layers.MultiHeadAttention(
        TRANSFORMER_HEADS, EMBED_DIM//TRANSFORMER_HEADS
    )(x, x)
    x = layers.LayerNormalization()(x + attn)

    ff = layers.TimeDistributed(layers.Dense(TRANSFORMER_FF, activation="relu"))(x)
    ff = layers.TimeDistributed(layers.Dense(EMBED_DIM))(ff)
    x = layers.LayerNormalization()(x + ff)

    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True))(x)

    w = layers.Dense(1)(x)
    w = tf.nn.softmax(w, axis=1)
    sym_feat = tf.reduce_sum(x * w, axis=1)

    # Medical history text
    inp1 = layers.Input((1,), dtype=tf.string)
    inp2 = layers.Input((1,), dtype=tf.string)
    inp3 = layers.Input((1,), dtype=tf.string)

    emb = layers.Embedding(MAX_TOKENS, TEXT_EMBED_DIM, mask_zero=True)

    mh1 = layers.GlobalAveragePooling1D()(emb(vect1(inp1)))
    mh2 = layers.GlobalAveragePooling1D()(emb(vect2(inp2)))
    mh3 = layers.GlobalAveragePooling1D()(emb(vect3(inp3)))

    # Effect flags
    inp_eff = layers.Input((3,))

    # Combine
    z = layers.Concatenate()([sym_feat, mh1, mh2, mh3, inp_eff])
    z = layers.Dense(DENSE_UNITS, activation="relu")(z)
    z = layers.Dropout(DROPOUT)(z)

    out_d = layers.Dense(num_classes_d, activation="softmax", name="disease")(z)
    out_s = layers.Dense(num_classes_s, activation="softmax", name="severity")(z)

    model = Model(
        [inp_sym, inp1, inp2, inp3, inp_eff],
        [out_d, out_s]
    )

    model.compile(
        optimizer=optimizers.Adam(LR),
        loss={
            "disease":"sparse_categorical_crossentropy",
            "severity":"sparse_categorical_crossentropy"
        },
        metrics={
            "disease":"accuracy",
            "severity":"accuracy"
        }
    )
    return model

model = build_model()
model.summary()

# ---------------------- Callbacks ----------------------
cb = [
    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(patience=4),
]

# ---------------------- Training ----------------------
history = model.fit(
    [X_sym_tr, mh1_tr, mh2_tr, mh3_tr, X_eff_tr],
    {"disease":y_d_tr, "severity":y_s_tr},
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb,
    verbose=2
)

# ---------------------- Evaluation ----------------------
pred_d, pred_s = model.predict([X_sym_te, mh1_te, mh2_te, mh3_te, X_eff_te])
pred_d = np.argmax(pred_d, axis=1)
pred_s = np.argmax(pred_s, axis=1)

print("\nDisease accuracy:", accuracy_score(y_d_te, pred_d))
print("Severity accuracy:", accuracy_score(y_s_te, pred_s))

# ---------------------- Confusion Matrices ----------------------
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_s_te, pred_s),
            annot=True, fmt="d",
            xticklabels=le_s.classes_,
            yticklabels=le_s.classes_)
plt.title("Severity Confusion Matrix")
plt.show()

# Disease (top 20)
top = pd.Series(y_d_te).value_counts().head(20).index
cm = confusion_matrix(y_d_te, pred_d)
cm = cm[np.ix_(top, top)]

plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="Reds")
plt.title("Disease Confusion Matrix (Top 20)")
plt.show()

# ---------------------- Training Curves ----------------------
plt.figure(figsize=(10,4))
plt.plot(history.history["disease_accuracy"], label="Train Disease Acc")
plt.plot(history.history["val_disease_accuracy"], label="Val Disease Acc")
plt.legend()
plt.title("Disease Accuracy Curve")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(history.history["severity_accuracy"], label="Train Severity Acc")
plt.plot(history.history["val_severity_accuracy"], label="Val Severity Acc")
plt.legend()
plt.title("Severity Accuracy Curve")
plt.show()

# ---------------------- Reports ----------------------
print("\nSeverity Report:\n",
      classification_report(y_s_te, pred_s, target_names=le_s.classes_))

