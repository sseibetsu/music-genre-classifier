import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_PATH = "data.json"
MODEL_PATH = "model.h5"

GENRE_MAPPING_REVERSE = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8,
    'rock': 9
}
def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return x, y


def prepare_datasets(test_size, validation_size):
    x, y = load_data(DATA_PATH)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)

    # 3D array
    x_train = x_train[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_model(input_shape):
    model = keras.Sequential()

    model.add(
        keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=input_shape))

    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))

    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))

    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))

    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dropout(0.5, noise_shape=None, seed=None))

    model.add(keras.layers.Dense(units=128, activation='relu'))

    model.add(keras.layers.Dense(units=10, activation='softmax'))

    return model


def predict(model, x, y):
    x = x[np.newaxis, ...]
    prediction = model.predict(x)

    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))


def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="lower right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == "__main__":
    x_train, x_validation, x_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    model = build_model(input_shape)

    # training the CNN
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=36, epochs=30)

    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print("Accuracy on test set is : {}".format(test_accuracy))

    # make prediction
    x = x_test[100]
    y = y_test[100]
    predict(model, x, y)

    plot_history(history)

    model.save('model.h5')