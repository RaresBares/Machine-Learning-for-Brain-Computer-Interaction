import torch
from torch import nn


class SoftmaxRegression(nn.Module):
    """
    Simple softmax regression (multinomial logistic regression) for frequency classification.

    This is a linear model: y = W @ x + b
    - W is a (num_classes, input_size) weight matrix
    - Each row of W is a weight vector for one class (One-vs-Rest)
    - Softmax is applied during loss calculation (CrossEntropyLoss)

    Args:
        input_size: Size of input features (default: 8*128 = 1024 for 8 EEG channels)
        num_classes: Number of frequency classes to predict
    """

    def __init__(self, input_size=8*128, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        """
        Forward pass - returns logits (unnormalized scores).

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        return self.linear(x)

    def predict(self, x):
        """
        Get class predictions with probabilities.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            predictions: Class indices (batch_size,)
            probabilities: Class probabilities (batch_size, num_classes)
        """
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities

    def get_weights(self):
        """
        Get the weight matrix for interpretation.

        Returns:
            W: Weight matrix of shape (num_classes, input_size)
            b: Bias vector of shape (num_classes,)
        """
        return self.linear.weight.data, self.linear.bias.data


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model for 5 frequency classes
    model = SoftmaxRegression(input_size=1024, num_classes=5).to(device)

    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 1024).to(device)

    # Get logits
    logits = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Get predictions
    predictions, probabilities = model.predict(x)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Sample probabilities: {probabilities[0]}")
