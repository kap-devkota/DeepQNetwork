from tensorflow import keras


def get_cnn(final_layer_hidden_size):
    model = keras.models.Sequential([
        keras.layers.Dense(24, input_shape=(4,), activation='relu'),
        keras.layers.Dense(24,  activation='relu'),
        keras.layers.Dense(2)
    ])

    model.compile(optimizer='Adam', loss=keras.losses.mean_squared_error)
    return model
