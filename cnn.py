from tensorflow import keras


def get_cnn(final_layer_hidden_size):

    model = keras.models.Sequential([
        keras.layers.Conv2D(16,
                            (8, 8),
                            strides=(4, 4),
                            padding='same',
                            input_shape=(84, 84, 4), activation='relu'),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(final_layer_hidden_size)
    ])

    model.compile(optimizer='Adam', loss=keras.losses.mean_squared_error)

    return model


callback = keras.callbacks.EarlyStopping(monitor='loss',
                                         min_delta=1,
                                         patience=1, verbose=0, mode='auto')