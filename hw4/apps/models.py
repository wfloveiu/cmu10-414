import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

def ConvBN(in_channels, out_channels, kernel_size, stride, device, dtype):
    return nn.Sequential(
        nn.Conv(in_channels, out_channels, kernel_size, stride=stride, bias=True, device=device, dtype=dtype),
        nn.BatchNorm2d(out_channels, device=device, dtype=dtype),
        nn.ReLU()
    )

# specially for 2 convBN layers 
class ResidualBlock(ndl.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, device, dtype):
        self.conv1 = ConvBN(in_channels,out_channels, kernel_size, stride, device, dtype)
        self.conv2 = ConvBN(in_channels,out_channels, kernel_size, stride, device, dtype)
    def forward(self, x):
        return self.conv2(self.conv1(x)) + x
        
class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.conv1 = ConvBN(3, 16, 7, 4, device=device, dtype=dtype)
        self.conv2 = ConvBN(16, 32, 3, 2, device=device, dtype=dtype)
        self.resblock1 = ResidualBlock(32, 32, 3, 1, device=device, dtype=dtype)
        self.conv3 = ConvBN(32, 64, 3, 2, device=device, dtype=dtype)
        self.conv4 = ConvBN(64, 128, 3, 2, device=device, dtype=dtype)
        self.resblock2 = ResidualBlock(128, 128, 3, 1, device=device, dtype=dtype)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128,10, device=device, dtype=dtype)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.resblock1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.resblock2(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        self.hidden_size = hidden_size
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        seq_len, bs = x.shape
        x_embedding = self.embedding(x)
        out, h = self.seq_model(x_embedding, h)
        out = out.reshape((seq_len*bs, self.hidden_size)) #flatten
        out = self.linear(out)
        return out, h


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)