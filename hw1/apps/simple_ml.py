"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.open(image_filesname, 'rb') as img_f:
        img_f.read(4) #skip magic number
        num_images = int.from_bytes(img_f.read(4), 'big') # stored by high(big) endian
        rows = int.from_bytes(img_f.read(4), 'big')
        cols = int.from_bytes(img_f.read(4), 'big')
        
        image_data = img_f.read(num_images * rows * cols)
        X = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32)
        X = X.reshape(num_images, rows * cols)
        X /= 255.0 # normalize to [0,1]
        
    with gzip.open(label_filename, 'rb') as lb_f:
        lb_f.read(4)
        num_labels = int.from_bytes(lb_f.read(4), 'big')
        
        lable_data = lb_f.read(num_labels)
        y = np.frombuffer(lable_data, dtype=np.uint8)
        
    return X, y


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    batch_size = Z.shape[0]
    G = ndl.log(ndl.exp(Z).sum(axes=(1,)))
    I = (Z * y_one_hot).sum(axes=(1,))
    loss = (G-I).sum()
    return loss / batch_size


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    batch_cnt =  X.shape[0] // batch
    num_class = W2.shape[1]
    one_hot_y = np.eye(num_class)[y]
    
    for batch_inx in range(batch_cnt):
        x_batch = X[batch_inx * batch : (batch_inx+1) * batch]
        y_batch = one_hot_y[batch_inx * batch : (batch_inx+1) * batch]
        x_tensor = ndl.Tensor(x_batch)
        y_tensor = ndl.Tensor(y_batch)
        Z = ndl.relu(x_tensor@W1) @ W2
        loss_err = softmax_loss(Z, y_tensor) #ndl.Tensor
        loss_err.backward() # 反向传播，求所有Tesor的梯度
        
        new_W1 = ndl.Tensor(W1.numpy() - lr*W1.grad.numpy())
        new_W2 = ndl.Tensor(W2.numpy() - lr*W2.grad.numpy())
        W1, W2 = new_W1, new_W2
    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
