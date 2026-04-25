import torch
import torch.nn as nn


# Inputs
# A 'tensor' is just a multidimensional matrix
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

# Targets
# Essentially holds the answers to the input 'X'
y_and = torch.tensor([[0.0], [0.0], [0.0], [1.0]])
y_or  = torch.tensor([[0.0], [1.0], [1.0], [1.0]])
y_xor = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


# ------------ PART D ---------------------------------
class SingleLayerNN(nn.Module):
    def __init__(self):
        # https://docs.python.org/2/library/functions.html#super
        # I don't exactly get it, but it seems common
        super().__init__()
        self.linear = nn.Linear(2, 1)  # 2 inputs → 1 output

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2)  # 2 → 2
        self.output = nn.Linear(2, 1)  # 2 → 1

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

def train(model, X, y, epochs=10000, learning_rate=0.1):
    criterion = nn.BCELoss()  # Binary Cross Entropy
    # Creates a gradient
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Stochastic Gradient Descent

    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        # Determine loss using BCE. Uses predicted output vs actual output to calculate loss
        loss = criterion(outputs, y)

        # Backward pass (also known as backpropogation)
        # https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html
        # Recommends we might need to zero gradient before using '.backword()'
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 2000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            print(model(X))

def part_d():
    model = XORNet()
    train(model, X, y_xor, epochs=10000, learning_rate=0.5)

    print("\nXOR Results:")
    print(model(X))
    print("\nXOR Results (rounded):")
    print(model(X).round())
# ------------ END PART D -----------------------------

# ------------ PART C ---------------------------------

# NAND as a neuron
# Forcing behavior
def NAND(a, b):
    return torch.sigmoid(-10*a - 10*b + 15)

def AND(a, b):
    return torch.sigmoid(10*a + 10*b - 15)

def OR(a, b):
    return torch.sigmoid(10*a + 10*b - 5)

def NOT(a):
    return torch.sigmoid(-10*a + 5)

# Performs: XOR = AND(OR(x,y), NOT(AND(x,y))
def XOR_mixed(x):
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]

    or_ = OR(x1, x2)
    and_ = AND(x1, x2)
    not_and = NOT(and_)

    return AND(or_, not_and)

# Performs: XOR = NAND(NAND(x,NAND(x,y)), NAND(y,NAND(x,y)))
# XOR made up of a bunch of NANDs
# def XOR_circuit(x):
#     x1 = x[:, 0:1]  # All rows, 1st column (at index 0) -> 0.0
#                     #                                      0.0
#                     #                                      1.0
#                     #                                      1.0
#
#     x2 = x[:, 1:2]  # All rows, 2nd column (at index 1)
#
#     neuron1 = NAND(x1, x2)      # Feed each row of 'X' into NAND
#     neuron2 = NAND(x1, neuron1)
#     neuron3 = NAND(x2, neuron1)
#
#     out = NAND(neuron2, neuron3)
#
#     return out
# ------------ END PART C -----------------------------

def main():
#    part_d()
    print(XOR_mixed(X))

if __name__ == "__main__":
    main()