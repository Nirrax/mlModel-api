import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt

DATA_PATH = "Data/data.json"

def load_data(data_path):
  
  print("Loading data")
  
  with open(data_path, "r") as fp:
    data = json.load(fp)
    
  X = np.array(data["mfcc"])
  y = np.array(data["labels"])
  
  return X, y

def prepare_datasets(test_size, validation_size):
 
  # load data
  X, y = load_data(DATA_PATH)
  
  # create train/test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
  
  # create train/validation split
  X_train, X_validation, y_train, y_validation = train_test_split(X_train,
                                                                  y_train,
                                                                  test_size=validation_size)
  
  # mapping 2d array on 3d array
  X_train = X_train[..., np.newaxis]
  X_validation = X_validation[..., np.newaxis]
  X_test = X_test[..., np.newaxis]
  
  return  X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):
  
  # create model
  model = keras.Sequential()
  
  # 1st conv layer
  model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
  model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
  model.add(keras.layers.BatchNormalization())
  
  # 2nd conv layer
  model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
  model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
  model.add(keras.layers.BatchNormalization())
  
  # 3rd conv layer
  model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
  model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
  model.add(keras.layers.BatchNormalization())
  
  # flatten the output
  model.add(keras.layers.Flatten())
  
  # add a dense layer
  model.add(keras.layers.Dense(64, activation='relu'))
  model.add(keras.layers.Dropout(0.3))
  
  # output layer
  model.add(keras.layers.Dense(10, activation='softmax'))
  
  return model

def predict(model, X, y):
  
  X = X[np.newaxis, ...]
  
  prediction = model.predict(X)
  
  # extract index with max value
  predicted_index = np.argmax(prediction, axis=1)
  print("Expected index: {}, Predicted index: {}".format(y, predicted_index))
  
def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == "__main__":
  
  # create train, validation and test sets
  X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)
  
  # build cnn net
  input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
  model = build_model(input_shape)
  
  # compile the network
  optimizer = keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  
  # train the model
  history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)
  model.save('CNN_Model.keras')
  
  # evaluate cnn on the test set
  test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
  train_error, train_accuracy = model.evaluate(X_train, y_train, verbose=1)
  print("Accuracy on train set: {}".format(train_accuracy))
  print("Accuracy on test set: {}".format(test_accuracy))
  
  plot_history(history)
  
  # prediction on a sample
  #X = X_test[100]
  #y = y_test[100]
  
  #predict(model, X, y)
  