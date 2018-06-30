from PIL import Image
import numpy as np
import pickle

def convert(image_path, saving_path):
    # converts a x pixel by x pixel image to 28*28
    basewidth = 28
    img = Image.open(image_path)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img = img.convert('1')
    img.save(saving_path)


def image_to_array(path, show_image=False, digit_black_background_white=False, cleaning_threshold=100):

    img = Image.open(path).convert('L')

    i  = np.array(img)
    if digit_black_background_white is False:
        i = 255-i
    i = i.flatten()
    if show_image:
        img.show()
        for a in range(len(i)):
            if a%28 == 0:
                print("")
            if i[a] > cleaning_threshold:
                print("@", end="")
            else:
                print(".", end="")
    i[np.where(i <= cleaning_threshold )] = 0
    return i


def predict(path,  show_image, digit_black_background_white, cleaning_threshold):
    file = open('checkpoints/checkpoint_3.pkl', 'rb')
    nn = pickle.load(file)
    file.close()
    import neural_net
    bnn = neural_net.NueralNetwork( 784, 30, 30, 10 )
    bnn.set_parameters(nn.W1, nn.W2, nn.W3, nn.b1, nn.b2, nn.b3)
    print(bnn.predict(image_to_array(path, show_image=show_image,
                                     digit_black_background_white=digit_black_background_white, cleaning_threshold=cleaning_threshold)))


def predict_custom_image(image_path , converted_image_saving_path,  show_image, digit_black_background_white, cleaning_threshold ):

    convert(image_path, converted_image_saving_path)
    predict(converted_image_saving_path,  show_image, digit_black_background_white, cleaning_threshold)
