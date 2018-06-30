import numpy as np
import sys
import time


class NueralNetwork():
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        self.input_layer_size = input_size
        self.hidden_layer1_size = hidden1_size
        self.hidden_layer2_size = hidden2_size
        self.output_layer_size = output_size
        self.data_loss = []
        np.random.seed(1)
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer1_size)
        self.W2 = np.random.randn(self.hidden_layer1_size, self.hidden_layer2_size)
        self.W3 = np.random.randn(self.hidden_layer2_size, self.output_layer_size)
        self.b1 = np.zeros((1, self.hidden_layer1_size))
        self.b2 = np.zeros((1, self.hidden_layer2_size))
        self.b3 = np.zeros((1, self.output_layer_size))

    def train(self, input_values, label, batch_size, epochs, learning_rate, error=False):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        print("training...")
        starting_time = time.time()
        N = len(input_values)
        for j in range(epochs):
            i = 0
            print("## Epoch: ", j+1, "/", epochs, " ##")
            print("Time Lapsed: ", time.time() - starting_time)
            if error:
                self.data_loss.append(self.error(input_values, label))
            while i+batch_size < N:
                sys.stdout.write("\rCompleted " + str((100*i/N)) + " in epoch " + str(j+1))
                sys.stdout.flush()

                X = input_values[i:i+batch_size]
                y = label[i:i+batch_size]
                i = i+batch_size

                z1 = X @ self.W1 + self.b1
                a1 = np.tanh(z1)
                # tanh activation function
                z2 = a1 @ self.W2 + self.b2
                a2 = np.tanh(z2)
                # tanh activation function
                z3 = a2 @ self.W3 + self.b3
                a3_temp = np.exp(z3)
                a3 = a3_temp / np.sum(a3_temp, keepdims=True, axis=1)
                # softmax activation function


                # cross entropy error function
                delta4 = a3
                delta4[range(len(y)), y] -= 1

                dw3 = a2.T @ delta4
                db3 = np.sum(delta4, axis=0, keepdims=True)

                delta3 = (delta4 @ self.W3.T) * (1 - np.power(a2, 2))
                dw2 = np.dot(a1.T, delta3)
                db2 = np.sum(delta3, axis=0)

                delta2 = (delta3 @ self.W2.T) * (1 - np.power(a1, 2))
                dw1 = np.dot(X.T, delta2)
                db1 = np.sum(delta2, axis=0)

                self.W1 += -learning_rate * dw1
                self.b1 += -learning_rate * db1
                self.W2 += -learning_rate * dw2
                self.b2 += -learning_rate * db2
                self.W3 += -learning_rate * dw3
                self.b3 += -learning_rate * db3

            print("")


    def error(self, X, y):
        # cross entropy error
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.tanh(z2)
        z3 = a2 @ self.W3 + self.b3
        a3_temp = np.exp(z3)
        a3 = a3_temp / np.sum(a3_temp, keepdims=True, axis=1)

        correct_loss = -np.log(a3[range(len(y)), y])
        data_loss = np.sum(correct_loss)
        data_loss = data_loss / len(X)
        print(data_loss)
        return data_loss

    def show_performance(self, X, y):

        print("Testing...")
        print("Error is: ", end="")

        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.tanh(z2)
        z3 = a2 @ self.W3 + self.b3
        a3_temp = np.exp(z3)
        a3 = a3_temp / np.sum(a3_temp, keepdims=True, axis=1)
        y_p = np.argmax(a3, axis=1)
        self.error(X, y)
        score = 0
        for i in range(len(y)):
            if y_p[i] == y[i]:
                score += 1
        print("Score is ", score/len(y))

    def plot_learning_curve(self):
        import matplotlib.pyplot as plt

        plt.plot( range(len(self.data_loss)), self.data_loss)
        plt.xlabel("No of epoch")
        plt.ylabel("Error")
        plt.title("Error vs epoch")
        plt.show()

    def set_parameters(self, w1, w2, w3, b1, b2, b3):

        self.W1 = w1
        self.W2 = w2
        self.W3 = w3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def predict(self, X):

        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.tanh(z2)
        z3 = a2 @ self.W3 + self.b3
        a3_temp = np.exp(z3)
        a3 = a3_temp / np.sum(a3_temp, keepdims=True, axis=1)
        y_p = np.argmax(a3, axis=1)
        # print(a2.shape)
        return y_p
