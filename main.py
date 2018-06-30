import mnist_initialization as initial
import pickle


def initial_training(batch_size=1, epochs=1000, learning_rate=0.000006):

    #######################
    checkpoint_no = 4

    import neural_net
    train_images, train_label, test_images, test_label = initial.load_data_mnist()
    nn = neural_net.NueralNetwork(784, 30, 30, 10)
    nn.train(train_images, train_label, batch_size, epochs, learning_rate, error=True)
    nn.show_performance(test_images, test_label)
    nn.plot_learning_curve()
    file = open('checkpoints/checkpoint_' + str(checkpoint_no), 'wb')
    pickle.dump(nn , file)
    file.close()

def updations(train=False):

    #######################
    checkpoint_no = 3

    file = open('checkpoints/checkpoint_' + str(checkpoint_no), 'rb')
    nn = pickle.load(file)
    file.close()


    import neural_net
    bnn = neural_net.NueralNetwork( 784, 30, 30, 10 )
    train_images, train_label, test_images, test_label = initial.load_data_mnist()
    # bnn.show_performance(test_images, test_label)
    bnn.set_parameters(nn.W1, nn.W2, nn.W3, nn.b1, nn.b2, nn.b3)


    if train:
        bnn.train(train_images, train_label, 1, 100, 0.000006, error=True)

    bnn.show_performance(train_images, train_label)
    bnn.show_performance(test_images, test_label)

def check_performance(X, y):

    #######################
    checkpoint_no = 3

    file = open('checkpoints/checkpoint_' + str(checkpoint_no), 'rb')
    nn = pickle.load(file)
    file.close()

    import neural_net
    bnn = neural_net.NueralNetwork(784, 30, 30, 10)
    # bnn.show_performance(test_images, test_label)
    bnn.set_parameters(nn.W1, nn.W2, nn.W3, nn.b1, nn.b2, nn.b3)

    print("Train data performance")
    bnn.show_performance(X, y)

def check_on_custom_image():
    image_path = ''
    converted_image_saving_path = ''
    show_image = False
    digit_black_background_white = False
    cleaning_threshold = 100

    # make sure that the input image is of z*z pixel, i.e square shaped
    # or else it will throw exception

    import custom_image
    custom_image.predict_custom_image(image_path, converted_image_saving_path, show_image, digit_black_background_white, cleaning_threshold)



if __name__ == '__main__':
    check_performance()