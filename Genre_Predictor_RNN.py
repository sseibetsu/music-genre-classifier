import json
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd

NEW_DATA_PATH = "new_data.json"

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

def predict(model, x):
    x = x[np.newaxis, ...]
    prediction = model.predict(x)

    predicted_index = np.argmax(prediction, axis=1)
    return predicted_index

def plot_history(history):
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="lower right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":
    x, y = load_data(NEW_DATA_PATH)

    model = keras.models.load_model('rnn_model.h5')

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    test_error, test_accuracy = model.evaluate(x, y, verbose=1)
    print("Accuracy on new data is : {}".format(test_accuracy))
    print("Error on new data is : {}".format(test_error))

    predictions = []
    for i in range(len(x)):
        prediction_index = predict(model, x[i])
        predictions.append(prediction_index)
    predictions_df = pd.DataFrame({'Predicted Index': predictions, 'Actual Index': y})
    print(predictions_df)

    history = model.fit(x, y, epochs=30, validation_split=0.2, batch_size=36, verbose=1)
    plot_history(history)
