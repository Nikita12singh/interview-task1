from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.data_preprocessing import load_and_clean_text, create_sequences
from src.model import build_model
from src.config import *
import os

def train():
    text = load_and_clean_text("data/shakespeare.txt")
    X, y, char_to_idx, idx_to_char = create_sequences(text, SEQ_LENGTH)

    model = build_model(
        vocab_size=len(char_to_idx),
        seq_length=SEQ_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS
    )

    os.makedirs("model", exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='loss', patience=3),
        ModelCheckpoint("model/text_generator.h5", save_best_only=True)
    ]

    model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )

if __name__ == "__main__":
    train()
