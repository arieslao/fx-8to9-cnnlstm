import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_lstm(seq_len, n_features, cnn_filters=32, cnn_kernel=3, lstm_units=64, dropout=0.2, lr=1e-3):
    inp = layers.Input(shape=(seq_len, n_features))
    x = layers.Conv1D(filters=cnn_filters, kernel_size=cnn_kernel, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(filters=cnn_filters, kernel_size=cnn_kernel, padding="causal", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model
