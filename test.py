import numpy as np
from network import Network
import mnist
import pickle
import function
import cv2

# load data
num_classes = 10
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print("Training...")

# data processing
X_test = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2]).astype('float32') #flatten 28x28 to 784x1 vectors, [60000, 784]
x_test = X_test / 255 #normalization
y_test = test_labels

with open('weights.pkl', 'rb') as handle:
    b = pickle.load(handle)

weight1 = b[0]
bias1 = b[1]
weight2 = b[2]
bias2 = b[3]

num = 0

while num < test_images.shape[0]:
    input_layer = np.dot(x_test[num:num+1], weight1)
    hidden_layer = function.relu(input_layer + bias1)
    scores = np.dot(hidden_layer, weight2) + bias2
    probs = function.softmax(scores)
    predict = np.argmax(probs)

    img = np.zeros([28,28,3])

    img[:,:,0] = test_images[num]
    img[:,:,1] = test_images[num]
    img[:,:,2] = test_images[num]


    resized_image = cv2.resize(img, (100, 100)) 
    cv2.putText(resized_image, str(predict), (5,20), cv2.FONT_HERSHEY_DUPLEX, .7, (0,255, 0), 1)
    cv2.imshow('input', resized_image)
    k = cv2.waitKey(0)
    if k==27:    # Esc key to stop
        break
    num += 1

cv2.destroyAllWindows()
