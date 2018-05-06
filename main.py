import numpy as np
from network import Network
import mnist


# load data
num_classes = 10
train_images = mnist.train_images() #[60000, 28, 28]
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print("Training...")

# data processing
X_train = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2]).astype('float32') #flatten 28x28 to 784x1 vectors, [60000, 784]
x_train = X_train / 255 #normalization
y_train = np.eye(num_classes)[train_labels] #convert label to one-hot

X_test = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2]).astype('float32') #flatten 28x28 to 784x1 vectors, [60000, 784]
x_test = X_test / 255 #normalization
y_test = test_labels

net = Network(
                 num_nodes_in_layers = [784, 20, 10], 
                 batch_size = 1,
                 num_epochs = 5,
                 learning_rate = 0.001, 
                 weights_file = 'weights.pkl'
             )

net.train(x_train, y_train)


print("Testing...")
net.test(x_test, y_test)
