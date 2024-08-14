import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    main = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    return nn.Sequential(nn.Residual(main), nn.ReLU())


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    layers = []
    layers.append(nn.Flatten())
    layers.append(nn.Linear(dim, hidden_dim))
    layers.append(nn.ReLU())
    for i in range(num_blocks):
        layers.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    layers.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*layers)


def epoch(dataloader, model, opt=None): # opt是优化器，如SGD， Adam
    np.random.seed(4)
    if opt is None:
        model.eval()
    else:
        model.train()
    loss_fuc = nn.SoftmaxLoss()
    acc_num = 0
    losses = []
    for X, y in dataloader:
        # print(X.shape)
        out = model(X)
        loss = loss_fuc(out, y)
        if opt is not None:
            loss.backward() #计算节点梯度
            opt.step() #权重更新
            
        losses.append(loss.numpy())
        acc_num += (out.numpy().argmax(axis=1) == y.numpy()).sum()

    
    return 1 - acc_num / len(dataloader.dataset), np.mean(losses)
        
            


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    #  Initialize tranning dataloader
    trainning_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    )
    trainning_data_loader = ndl.data.DataLoader(trainning_dataset, batch_size, shuffle=True)
    #  Initialize test dataloader
    test_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    )
    test_data_loader = ndl.data.DataLoader(test_dataset, batch_size)

    shape = test_data_loader.dataset.images.shape
    dim = shape[1] * shape[2]
    print(dim)
    model = MLPResNet(dim=dim, hidden_dim=hidden_dim)

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_err,train_loss = 0, 0
    test_err,test_loss = 0, 0

    for i in range(epochs):
        train_err, train_loss = epoch(trainning_data_loader, model, opt)
        print("Epoch %d: Train err: %f, Train loss: %f" % (
            i, train_err, train_loss
        ))
    test_err, test_loss = epoch(test_data_loader, model)
    return train_err, train_loss, test_err, test_loss



if __name__ == "__main__":
    train_mnist(data_dir="../data")


