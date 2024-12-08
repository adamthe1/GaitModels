import torch
import torch.nn as nn

class MEDLoss(nn.Module):  # Mean Euclidean Distance Loss
    def __init__(self, epsilon=1e-6):
        """
        Initializes the MEDLoss module.

        Args:
            epsilon (float): Small value to prevent division by zero or sqrt of zero.
        """
        super(MEDLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predicted, target):
        """
        Forward pass to compute the Mean Euclidean Distance Loss.

        Args:
            predicted (torch.Tensor): Predicted tensor of shape [B, 32, 3].
            target (torch.Tensor): Ground truth tensor of shape [B, 32, 3].

        Returns:
            torch.Tensor: Scalar loss value representing the mean Euclidean distance.
        """
        # Calculate the difference
        diff = predicted - target  # Shape: [B, 32, 3]

        # Compute Euclidean distance for each element
        # Using torch.norm for clarity and efficiency
        # Specify dim=2 to compute norm over the last dimension (channels)
        euclidean_distance = torch.norm(diff, p=2, dim=2)  # Shape: [B, 32]

        # Optionally, add epsilon inside the norm to prevent sqrt(0)
        # Uncomment the following line if necessary
        # euclidean_distance = torch.sqrt(torch.sum(diff ** 2, dim=2) + self.epsilon)

        # Compute the mean Euclidean distance over all elements in the batch
        loss = torch.mean(euclidean_distance)  # Scalar

        return loss

def entropy_regularization(indices, codebook_size):
    indices = indices.view(-1)

    N = indices.numel()  # or len(indices)
    # Compute the usage counts
    usage_counts = torch.bincount(indices, minlength=codebook_size)
    # Calculate the probabilities correctly by dividing by N, not codebook_size
    codebook_usage_probs = usage_counts.float() / N
    # Compute entropy
    entropy = -torch.sum(codebook_usage_probs * torch.log(codebook_usage_probs + 1e-8))

    # Compute maximum entropy correctly
    max_entropy = torch.log(torch.tensor(codebook_size, dtype=torch.float32))

    # Return the normalized entropy
    return 1 - (entropy / max_entropy)

def avg_individual_entropy(batch_indices):
    # batch indices is a tensor of shape (batch_size, seq_len)
    size_seq = batch_indices.size(1)
    entropy_sum = 0
    for i in range(batch_indices.size(0)):
        indices = batch_indices[i]
        N = indices.numel()  # or len(indices)
        # Compute the usage counts
        usage_counts = torch.bincount(indices, minlength=size_seq)
        # Calculate the probabilities correctly by dividing by N, not codebook_size
        token_usage = usage_counts.float() / N
        # Compute entropy
        entropy = -torch.sum(token_usage * torch.log(token_usage + 1e-8))

        # Compute maximum entropy correctly
        max_entropy = torch.log(torch.tensor(size_seq, dtype=torch.float32))

        entropy_sum += 1 - (entropy / max_entropy)

    # Return the normalized entropy
    return entropy_sum / len(batch_indices)

if __name__ == "__main__":
    codebook_size = 2
    seq = torch.tensor([1, 1, 2, 3, 0, 5, 5, 0, 1, 1, 2, 2])
    seq = seq[::2]
    print(seq)
    a = 0
    for i in range(0, len(seq), 2):
        indices = seq[i:i+2]
        print(indices)
        a += entropy_regularization(indices, codebook_size)
        print("Exact match", entropy_regularization(indices, codebook_size))  # Expected output: 0
    print(a/3)

    indices = torch.tensor([1] * 8192 + [0] * 8192)
    print("All same", entropy_regularization(indices, codebook_size))  # Expected output: 0

    indices = torch.tensor([1] * 32 + [0] * 32)
    print("Double size", entropy_regularization(indices, codebook_size))  # Expected output: 0

    indices = torch.tensor(list(range(16392)))
    print("max entropy", entropy_regularization(indices, codebook_size))  # Expected output: 0
