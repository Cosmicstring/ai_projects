import torch

# Implement a simple nn mapping sigmoid((X*W) + bias)
# with X, W \in R^{1 x 10}

torch.manual_seed(42)

features = torch.randn((1,10))
weights = torch.randn_like(features)
bias = torch.randn((1,1))

print(torch.sigmoid((features*weights).sum() + bias))

# Matrix multiplications more efficient, especially for GPUs
wtranspose = weights.view(10,1)
print(torch.sigmoid(torch.mm(features, wtranspose).sum() + bias))
# torch.matmul supports broadcasts
print(torch.sigmoid(torch.matmul(features, wtranspose).sum() + bias))

#-----------------------------------

# Using a different configuration with one 2 dimensional hidden layer
n_input = features.shape[1]
n_hidden = 2
n_output = 1

# Weights for input -> hidden
W1 = torch.randn((n_input, n_hidden))

# Weights for hidden -> output
W2 = torch.randn((n_hidden, n_output))

# Biases for the input -> hidden
B1 = torch.randn((1, n_hidden))

# Biases for the hidden -> output
B2 = torch.randn((1, n_output))

print(torch.sigmoid(
    torch.mm(
        torch.sigmoid(torch.mm(features, W1) + B1), W2) + B2))

#-----------------------------------

# Copying over to numpy with:
#       torch.numpy()
# Copying over from numpy:
#       torch.from_numpy()

# The pointers are passed from numpy to pytorch,
# hence if the entries change in one of the two changes them in both

