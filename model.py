import os
import numpy as np
from sklearn.model_selection import train_test_split

# import tensorflow
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define the EarlyStopping and ReduceLROnPlateau callbacks
early_stopping = EarlyStopping(
    monitor="val_loss", patience=20, verbose=1, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, verbose=1, min_lr=1e-4
)


def lstm_model():
    # Set the path to the Datasets directory
    datasets_dir = "Datasets/"

    # Load the x and y data
    x = np.load(datasets_dir + "x.npy")
    y = np.load(datasets_dir + "y.npy")

    # Load the label map
    label_map = np.load(
        os.path.join(datasets_dir, "label_map.npy"), allow_pickle=True
    ).item()

    # Apply train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Reshape the input data to have the shape (batch_size, 30, 21 * 3)
    x_train = x_train.reshape(x_train.shape[0], 30, -1)
    x_test = x_test.reshape(x_test.shape[0], 30, -1)
    print(x_train.shape)

    # Define the LSTM model
    model = Sequential()
    # model.add(B)
    model.add(
        Bidirectional(
            LSTM(
                64,
                return_sequences=True,
                activation="relu",
                input_shape=(x_train.shape[1], x_train.shape[2]),
            )
        )
    )
    model.add(Bidirectional(LSTM(128, return_sequences=True, activation="relu")))
    model.add(Bidirectional(LSTM(64, return_sequences=False, activation="relu")))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    # Compile the model
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        epochs=40,
        batch_size=32,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, reduce_lr],
    )

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    # Save the model
    model.save("hand_gesture_model_bidirectional.h5")


lstm_model()
