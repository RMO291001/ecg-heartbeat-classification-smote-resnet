from tensorflow.keras import layers, models

def residual_block(x, filters=32, kernel_size=5):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    x = layers.MaxPool1D(pool_size=5, strides=2, padding="same")(x)
    return x

def build_kachuee_cnn(input_shape=(187, 1), n_classes=5):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 5, padding="same")(inputs)
    x = layers.ReLU()(x)
    for _ in range(5):
        x = residual_block(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)
