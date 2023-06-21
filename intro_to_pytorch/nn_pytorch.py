import torch

from torch import nn, Tensor
from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from collections import OrderedDict

import helper

SHOW = 0
torch.manual_seed(42)

# Set double precision
torch.set_default_dtype(torch.float64)

def softmax(input: Tensor):

    tmp = input.exp()

    tmpnorm = tmp.sum(dim=1)
    return tmp/tmpnorm.resize_((tmpnorm.shape[0],1))

class RandomLinearModel(nn.Module):

    def __init__(self,
                 input_layer: int,
                 hidden_layer: int,
                 output_layer: int) -> None:

        super().__init__()
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

        self.W1 = torch.randn((self.input_layer, self.hidden_layer))
        self.B1 = torch.randn((self.hidden_layer,))
        self.W2 = torch.randn((self.hidden_layer, self.output_layer))
        self.B2 = torch.randn((self.output_layer,))

    def forward(self, inputT: Tensor) -> Tensor:
        x = torch.sigmoid(torch.mm(inputT, self.W1) + self.B1)
        x = softmax(torch.mm(x, self.W2) + self.B2)
        return x

    def extra_repr(self) -> str:
        strout = f'in_features={self.input_layer}, out_features={self.output_layer}\n'
        strout += f'W1={self.W1} \n , W2={self.W2}\n'
        strout += f'B1={self.B1.numpy()} \n, B2={self.B2.numpy()}'
        return strout

class SimpleNetwork(nn.Module):

    def __init__(self) -> None:

        super().__init__()
        input_layer = 784
        hidden_layer1 = 128
        hidden_layer2 = 64
        output_layer = 10

        self.fc1 = nn.Linear(input_layer, hidden_layer1) # input -> hidden 1
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2) # hidden 1 -> hidden 2
        self.fc3 = nn.Linear(hidden_layer2, output_layer) # hidden 2 -> output
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.softmax(self.fc3(x), dim=-1)

        return out

def test_SimpleNetwork(input):

    model = SimpleNetwork()
    # It seems the parameters of the Linear model layer are randomly initialized
    print(model.fc1.weight)
    print(model.fc1.bias)

    ps = model.forward(input)

    helper.view_classify(input.view(1, 28, 28), ps)
    plt.show()

def test_RandomModel(input):

    input_layer = 784 # (28 x 28)
    hidden_layer = 256 #
    output_layer = 10 # 10 number categories (0..9)

    model = RandomLinearModel(input_layer,hidden_layer,output_layer)
    print(model.extra_repr())
    # Must resize to (1, 784) for the transpose
    input.resize_(1,784)
    prob1 = model.forward(input)
    print(prob1.shape)
    print(torch.all(prob1.sum(dim=1)))

def test_SequentialModel_training(training_set):

    input_layer = 784
    hidden_layers = [128,64]
    output_layer = 10
    EPOCHS = 5

    layers = OrderedDict(
        [
            ('fc1', nn.Linear(input_layer, hidden_layers[0])),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(hidden_layers[1], output_layer)),
            ('LogSoftmax', nn.LogSoftmax(dim=-1))
        ]
    )
    model = nn.Sequential(layers)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # The LogSoftmax is used for the CrossEntropyLoss, hence we stop at the output of `fc3` for feedin
    # into the LogSoftmax

    loss = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        _loss = 0
        for images, labels in training_set:
            # Flatten images
            images = images.view(images.shape[0], -1)

            scores = model(images)
            optimizer.zero_grad()
            _loss = loss(scores, labels)
            _loss.backward()
            optimizer.step()

        else:
            print("Current loss: ", _loss)


    for images, labels in training_set:
        img = images[42].view(1, 784)
        # Turn off gradients to speed up this part
        with torch.no_grad():
            logps = model(img)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            helper.view_classify(img.view(1, 28, 28), ps)
            plt.show()
        break

def test_LinearModel_FashionMNIST(trainloader, testloader,
                                  optim_kywrd="SGD",
                                  EPOCHS = 30):

    def get_test_acc():

        with torch.no_grad():
            images,labels = next(iter(testloader))

            # Force model to eval mode to correctly evaluate on test dataset
            model.eval()
            probs = torch.exp(model(images.view(images.shape[0],-1)))

            top_p, top_class = probs.topk(1, dim=1)
            matching = top_class == labels.view(*top_class.shape)
            acc = matching.sum()/matching.size().numel()
            print(f"{acc.item()*100}%\n\n")

    input_layer = 784
    hidden_layers = [128,64]
    output_layer = 10

    # Test with dropout:
    #   eval --> forces droput prob. to 0
    #   train --> turns dropout back on

    layers = OrderedDict(
        [
            ('fc1', nn.Linear(input_layer, hidden_layers[0])),
            ('relu1', nn.ReLU()),
            ('droput1', nn.Dropout(p=0.15)),
            ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
            ('relu2', nn.ReLU()),
            ('droput2', nn.Dropout(p=0.15)),
            ('fc3', nn.Linear(hidden_layers[1], output_layer)),
            ('LogSoftmax', nn.LogSoftmax(dim=-1))
        ]
    )
    model = nn.Sequential(layers)

    # Let's also try different optimizers -- SGD, Adam, LBFGS, ConjugateGradient
    SGDoptimizer = optim.SGD(model.parameters(), lr=0.01)
    LFBGSoptimizer = optim.LBFGS(model.parameters(), history_size=20, max_iter=4) # LBFGS uses 2nd order information, so allows for bigger learning rate!
    ADAMoptimizer = optim.Adam(model.parameters(), lr=0.01) # Does Adam need smaller learning rate?

    loss = nn.CrossEntropyLoss()

    optimizer = None
    if optim_kywrd == "SGD":
        optimizer = SGDoptimizer
    if optim_kywrd == "Adam":
        optimizer = ADAMoptimizer
    if optim_kywrd == "LBFGS":
        optimizer = LFBGSoptimizer

    assert optimizer != None

    for epoch in range(EPOCHS):
        _loss = 0

        print(f"Test accuracy before epoch {epoch}:")
        get_test_acc()
        model.train()
        for images, labels in trainloader:
            # Flatten images
            images = images.view(images.shape[0], -1)

            if optim_kywrd == "LBFGS":
                def closure():
                    optimizer.zero_grad()
                    scores = model(images)
                    _loss = loss(scores, labels)
                    _loss.backward()
                    return _loss
                optimizer.step(closure)
                scores = model(images)
                _loss = loss(scores, labels)
            else:
                scores = model(images)
                optimizer.zero_grad()
                _loss = loss(scores, labels)
                _loss.backward()
                optimizer.step()
        else:
            print("Current loss: ", _loss)

        print(f"Test accuracy after epoch {epoch}:")
        get_test_acc()

    for images, labels in testloader:
        img = images[42].view(1, 784)
        # Turn off gradients to speed up this part
        with torch.no_grad():
            logps = model(img)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            helper.view_classify(img.view(1, 28, 28), ps, version="Fashion")
            plt.show()
        break


if __name__ == "__main__":

    #-------- Compose transforms to be performed on input dataset
    transform = [transforms.ToTensor(),
                 transforms.Normalize((0.5,), (0.5,)),
                ]
    transform = transforms.Compose(transform)

    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    print("Feature tensor shape:", images.shape)
    if SHOW:
        plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
        plt.show()

    # input_layer.size <- 28 x 28 == 784
    images = images.view(64,784) # or could put -1 for the last argument, automatically inferred
    input = images[0]

    #-------- Simple hand made linear mappings for the NN

    #test_RandomModel(input)

    #------- Fully utilizing the torch.nn modules

    #test_SimpleNetwork(input)

    #-------- Let's do the training of the model, using Sequential

    #test_SequentialModel_training(trainloader)

    #-------- Let's try to apply the same model to the Fashion-MNIST dataset

    # Download and load the training data
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    test_LinearModel_FashionMNIST(trainloader, testloader, optim_kywrd="Adam")
