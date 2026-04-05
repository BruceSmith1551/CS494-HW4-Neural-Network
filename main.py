import torch
import torch.nn as nn

class SingleLayerNN(nn.Module):
    def __init__(self):
        # https://docs.python.org/2/library/functions.html#super
        # I don't exactly get it, but it seems common
        super().__init__()
        self.linear = nn.Linear(2, 1)  # 2 inputs → 1 output

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

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
        # Always print results
        elif epoch == epochs:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


def main():
    model_and = SingleLayerNN()
    train(model_and, X, y_and)

    print("\nAND Results:")
    print(model_and(X))
    print("\nAND Results (rounded):")
    print(model_and(X).round())


if __name__ == "__main__":
    main()