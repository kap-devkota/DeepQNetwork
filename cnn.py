from tensorflow import keras


def get_cnn(final_layer_hidden_size):
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same',
                            input_shape=(84, 84, 4)),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(32, (3, 3), padding='same'),
        keras.layers.Activation('softmax'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(rate=0.25),

        keras.layers.Conv2D(32, (3, 3), padding='same'),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(32, (3, 3), padding='same'),
        keras.layers.Activation('softmax'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(rate=0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(512),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(rate=0.25),
        keras.layers.Dense(final_layer_hidden_size),
        keras.layers.Activation('softmax')
    ])

    model.compile(optimizer='Adam', loss=keras.losses.mean_squared_error)
    return model
