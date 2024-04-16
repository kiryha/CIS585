"""
## Recognizer Intro
Build a neuron network for handwritten digit recognition using existing labeled data sets.
Data source: www.kaggle.com/c/digit-recognizer
Data explanation: www.wikiwand.com/en/mnist_database

## Training Data
Source data is a set of 28x28 pixels images of handwritten digits (784 pixel values)
CSV files rows: digits tata (0, 1, 2, 3 ... 9), 42 000 digit examples
CSV file columns: label (digit), pixel 0, pixel 1, ... pixel 783

## Process
3 parts of training:
    - forward propagation
    - backward propagation
    - update parameters
"""


import os
import numpy as np
import pandas as pd
from PySide2 import QtWidgets, QtGui
from PIL import Image, ImageOps
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ui import ui_main


class MatplotlibWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.canvas)

    def update_plot(self, image):

        self.ax.clear()
        self.ax.imshow(image, interpolation='nearest')
        self.ax.axis('off')
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)  # remove margins
        self.canvas.draw()


class Recognizer(QtWidgets.QMainWindow, ui_main.Ui_Recognizer):
    def __init__(self):
        super(Recognizer, self).__init__()
        self.setupUi(self)

        # Setup plot image display
        self.plot_widget = MatplotlibWidget(self)
        self.layImages.addWidget(self.plot_widget)
        self.display_random()

        # Data
        self.custom_image_path = None
        self.numbers_data_test = None
        self.data_display = None

        # Model
        self.W1_path_source = f'{root}/data/model/s_W1.csv'
        self.W2_path_source = f'{root}/data/model/s_W2.csv'
        self.b1_path_source = f'{root}/data/model/s_b1.csv'
        self.b2_path_source = f'{root}/data/model/s_b2.csv'
        self.W1_path_extended = f'{root}/data/model/e_W1.csv'
        self.W2_path_extended = f'{root}/data/model/e_W2.csv'
        self.b1_path_extended = f'{root}/data/model/e_b1.csv'
        self.b2_path_extended = f'{root}/data/model/e_b2.csv'
        self.W1 = None  # Weight matrix for the first layer
        self.b1 = None  # Bias vector for the first layer
        self.W2 = None  # Weight matrix for the second layer
        self.b2 = None  # Bias vector for the second layer

        # Load existing data and model
        self.load_model()

        # UI calls
        self.btnLoadImage.clicked.connect(self.load_image)
        self.btnExtendData.clicked.connect(self.extend_source_data)
        self.btnTrainSource.clicked.connect(self.train_source_model)
        self.btnTrainExtended.clicked.connect(self.train_extended_model)
        self.btnToSourceModel.clicked.connect(self.set_source_model)
        self.btnToExtendedModel.clicked.connect(self.set_extended_model)
        self.btnRecognize.clicked.connect(self.recognize)
        self.btnDisplay.clicked.connect(self.display_mnist)

    def display_random(self):

        # Show random noise in UI
        current_image = np.random.rand(100, 100)
        self.update_plot(current_image)

    def load_model(self):
        """
        Load trained model (source or extended) if it exists
        """

        print('Checking existing models...')

        # Load data for debug display
        data_file_extended = f"{root}/data/mnist/train.csv"
        if os.path.exists(data_file_extended):
            data_extended = np.array(pd.read_csv(data_file_extended))
            self.data_display = np.array(data_extended).T

        # Load test data
        data_file_test = f"{root}/data/mnist/test.csv"
        data_test = pd.read_csv(data_file_test)
        self.numbers_data_test = np.array(data_test).T / 255.

        # Load existing model, extended has priority
        if self.set_extended_model():
            return
        else:
            print('Extended model does not exists!')

        if not self.set_source_model():
            print('Source model does not exists! Train your model first!')
            self.linModel.setText('Not Exists')

    def extend_source_data(self):
        """
        Extend source data set with rotated images
        """

        print('Extending Source Data...')

        # Load the dataset
        input_csv = f"{root}/data/mnist/train.csv"
        output_csv = f"{root}/data/mnist/train_extended.csv"
        source_data = pd.read_csv(input_csv)

        # Iterate through each row in the dataset
        extended_data = []
        for index, row in source_data.iterrows():

            label = row[0]
            pixels = row[1:].values

            # Convert the pixels to an image (28x28)
            image = Image.fromarray(pixels.reshape(28, 28).astype('uint8'))

            # Original image
            extended_data.append([label] + list(pixels))

            # Rotate and add to extended data
            for angle in [90, 180, 270]:
                rotated_image = image.rotate(angle)
                rotated_pixels = np.array(rotated_image).flatten()
                extended_data.append([label] + list(rotated_pixels))

        # Convert extended data to DataFrame
        extended_df = pd.DataFrame(extended_data, columns=source_data.columns)

        # Save extended data to CSV, preserving header
        extended_df.to_csv(output_csv, index=False)

        print('Extended Data saved to train_extended.csv!')

    # Image Display
    def update_plot(self, image):

        self.plot_widget.update_plot(image)

    # ML functions
    def init_parameters(self):

        W1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5

        return W1, b1, W2, b2

    def rel_u(self, Z):

        return np.maximum(Z, 0)

    def rel_u_derivative(self, Z):

        return Z > 0

    def softmax(self, Z):

        A = np.exp(Z) / sum(np.exp(Z))

        return A

    def one_hot(self, Y):

        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T

        return one_hot_Y

    def get_predictions(self, A2):
        """
        Generates predicted labels (or outputs) based on the input data and the learned parameters of the model.

        This function converts the output of the network
        (which are probabilities in the case of a classification problem) into actual class predictions
        by taking the class with the highest probability for each sample.
        """

        return np.argmax(A2, 0)

    def get_accuracy(self, predictions, digit_labels):

        accuracy = np.sum(predictions == digit_labels) / digit_labels.size

        return accuracy

    # Propagation
    def forward_propagation(self, W1, b1, W2, b2, X):
        """
        Performs a forward propagation pass,
        which computes the output of the model given the current parameters and the input data.

        The forward propagation is the process by which the neural network uses the input data
        and the learned parameters to compute its output
        """

        Z1 = W1.dot(X) + b1  # Result of the linear transformation of the input layer
        A1 = self.rel_u(Z1)  # Result of applying the activation function to Z1
        Z2 = W2.dot(A1) + b2  # Result of the linear transformation of the hidden layer (A1)
        A2 = self.softmax(Z2)  # Result of applying the activation function to Z2

        return Z1, A1, Z2, A2

    def backward_propagation(self, Z1, A1, Z2, A2, W1, W2, X, Y):
        """
        Performs a backward propagation pass,
        which computes the gradients of the loss function with respect to the parameters.
        """

        m = Y.size
        one_hot_Y = self.one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * self.rel_u_derivative(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)

        return dW1, db1, dW2, db2

    def update_parameters(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        """
        Updates the parameters in the direction that decreases the loss function.
        The size of the update is controlled by the learning rate alpha.
        """

        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2

        return W1, b1, W2, b2

    def gradient_descent(self, alpha, iterations, digit_data, digit_labels):
        """
        Train neural network model by updating its parameters (weights and biases) iteratively
        to minimize the loss function (difference between the model's predictions and the actual values).
        """

        W1, b1, W2, b2 = self.init_parameters()  # The trained weights and biases of the model

        accuracy = 0
        for index in range(iterations):
            Z1, A1, Z2, A2 = self.forward_propagation(W1, b1, W2, b2, digit_data)
            dW1, db1, dW2, db2 = self.backward_propagation(Z1, A1, Z2, A2, W1, W2, digit_data, digit_labels)
            W1, b1, W2, b2 = self.update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

            if index % 10 == 0:
                print(f"Current iteration: {index}")
                predictions = self.get_predictions(A2)
                accuracy = self.get_accuracy(predictions, digit_labels)

        print(f'Model Accuracy: {accuracy}')

        return W1, b1, W2, b2

    # Recognition
    def make_predictions(self, X, W1, b1, W2, b2):
        """
        Uses the trained weights and biases to make predictions on a given input dataset.
        """

        _, _, _, A2 = self.forward_propagation(W1, b1, W2, b2, X)
        predictions = self.get_predictions(A2)

        return predictions

    def recognize_custom(self):
        """
        Recognize custom JPG
        """

        if not self.custom_image_path:
            self.statusbar.showMessage('Load custom Image first!')
            return

        image_label = os.path.basename(self.custom_image_path).replace('.jpg', '')

        # Convert image to proper array
        custom_image = Image.open(self.custom_image_path)
        custom_image = ImageOps.grayscale(custom_image)
        custom_image = custom_image.resize((28, 28))
        # Convert to numpy array and normalize
        custom_image = np.array(custom_image) / 255.
        # Flatten and reshape
        custom_image = custom_image.flatten().reshape(-1, 1)  # result is of shape (784, 1)

        # Recognize custom image
        prediction = self.make_predictions(custom_image, self.W1, self.b1, self.W2, self.b2)

        # Report
        message = f'Custom Number {image_label} recognized as {prediction[0]}'
        print(message)
        self.statusbar.showMessage(message)

    def display_mnist(self):
        """
        Load train.csv data into UI by row index
        index 41999 is number 9, which is row 42001 in train.csv
        """

        image_index = int(self.linIndex.text())

        current_image = self.data_display[1:, image_index, None]
        current_image = current_image.reshape((28, 28)) * 255
        self.update_plot(current_image)

        # Report
        message = f'Display Number {self.data_display[0][image_index]}'
        self.statusbar.showMessage(message)
        print(message)

    def recognize_mnist(self, index):
        """
        Recognize image from test.csv set by image data index
        """

        # Get image data from MNIST
        current_image = self.numbers_data_test[:, index, None]
        prediction = self.make_predictions(self.numbers_data_test[:, index, None], self.W1, self.b1, self.W2, self.b2)

        # Report
        message = f'Current image recognized as {prediction[0]}'
        self.statusbar.showMessage(message)
        print(message)

        # Show image in UI
        current_image = current_image.reshape((28, 28)) * 255
        self.update_plot(current_image)

    # UI calls
    def set_source_model(self):

        if os.path.exists(self.W1_path_source):
            print('Loading source model...')

            self.W1 = np.loadtxt(self.W1_path_source, delimiter=',')
            self.W2 = np.loadtxt(self.W2_path_source, delimiter=',')
            self.b1 = np.loadtxt(self.b1_path_source, delimiter=',')
            self.b2 = np.loadtxt(self.b2_path_source, delimiter=',')
            self.linModel.setText('Source')

            print('Source model loaded!')

            return True

    def set_extended_model(self):

        if os.path.exists(self.W1_path_extended):
            print('Loading extended model...')

            self.W1 = np.loadtxt(self.W1_path_extended, delimiter=',')
            self.W2 = np.loadtxt(self.W2_path_extended, delimiter=',')
            self.b1 = np.loadtxt(self.b1_path_extended, delimiter=',')
            self.b2 = np.loadtxt(self.b2_path_extended, delimiter=',')
            self.linModel.setText('Extended')

            print('Extended model loaded!')

            return True

    def load_image(self):
        """
        Load custom jpg
        """

        custom_image_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Custom File', f'{root}/data/custom_images/', '*.jpg')[0]

        if not custom_image_path:
            self.custom_image_path = None
            self.display_random()
            return

        self.custom_image_path = custom_image_path
        self.update_plot(np.array(Image.open(self.custom_image_path)))

    def train_source_model(self):
        """
        Train model on source data set
        """

        self.statusbar.showMessage('Training source model...')

        # Load data
        data_file_source = f"{root}/data/mnist/train.csv"
        data_source = np.array(pd.read_csv(data_file_source))

        rows_source, columns_source = data_source.shape
        data_source = data_source.T

        digit_labels_source = data_source[0]
        digit_data_source = data_source[1:columns_source]
        digit_data_source = digit_data_source / 255.

        alfa = float(self.linAlfa.text())
        iterations = int(self.linIterations.text())

        self.W1, self.b1, self.W2, self.b2 = self.gradient_descent(alfa, iterations, digit_data_source, digit_labels_source)

        # Save data to CSV files
        np.savetxt(self.W1_path_source, self.W1, delimiter=',')
        np.savetxt(self.W2_path_source, self.W2, delimiter=',')
        np.savetxt(self.b1_path_source, self.b1, delimiter=',')
        np.savetxt(self.b2_path_source, self.b2, delimiter=',')

        self.linModel.setText('Source')

        self.statusbar.showMessage('Model trained on source data and saved to files!')

    def train_extended_model(self):
        """
        Train model on extended data set (rotate all source digits 90, 180, 270 degrees)
        """

        self.statusbar.showMessage('Training extended model...')

        # Load data
        data_file_extended = f"{root}/data/mnist/train.csv"
        if not os.path.exists(data_file_extended):
            print('Extended data set not exists! Extend source data first.')
            return

        data_extended = np.array(pd.read_csv(data_file_extended))

        rows_extended, columns_extended = data_extended.shape
        data_extended = data_extended.T

        digit_labels_extended = data_extended[0]
        digit_data_extended = data_extended[1:columns_extended]
        digit_data_extended = digit_data_extended / 255.

        alfa = float(self.linAlfa.text())
        iterations = int(self.linIterations.text())

        self.W1, self.b1, self.W2, self.b2 = self.gradient_descent(alfa, iterations, digit_data_extended, digit_labels_extended)

        # Save data to CSV files
        np.savetxt(self.W1_path_extended, self.W1, delimiter=',')
        np.savetxt(self.W2_path_extended, self.W2, delimiter=',')
        np.savetxt(self.b1_path_extended, self.b1, delimiter=',')
        np.savetxt(self.b2_path_extended, self.b2, delimiter=',')

        self.linModel.setText('Extended')

        self.statusbar.showMessage('Model trained on source data and saved to files!')

    def recognize(self):

        if self.custom_image_path:
            # Recognize custom image
            self.recognize_custom()
        else:
            # Recognize image from MNIST
            self.recognize_mnist(int(self.linIndex.text()))


if __name__ == "__main__":

    root = os.path.dirname(os.path.abspath(__file__))

    app = QtWidgets.QApplication([])
    recognizer = Recognizer()
    recognizer.setWindowIcon(QtGui.QIcon('{0}/icons/recognizer.ico'.format(root)))
    recognizer.show()
    app.exec_()
