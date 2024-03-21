import torch
import numpy as np



class mixup:
    def __init__(self, alpha, sampling_method):
        self.alpha = alpha # mixup parameter
        self.sampling_method = sampling_method # either 1 or 2

    def mixup_data(self, x, y):
        """
        Apply mixup augmentation to a batch of inputs and labels.

        Args:
            x (Tensor): Batch of input images.
            y (Tensor): Batch of labels.

        Returns:
            mixed_x (Tensor): Mixed batch of input images.
            mixed_y (Tensor): Mixed batch of output labels.

        """
        batch_size = x.size(0)

        if self.sampling_method==1:
            # sample from beta distribution
            lam = np.random.beta(self.alpha, self.alpha)
        elif self.sampling_method==2:
            # sample from uniformly distribution
            lam = np.random.uniform(0,0.5)

        index = torch.randperm(batch_size)
        # linear interpolation
        mixed_x = lam * x + (1 - lam) * x[index, :]
        # convet y and y[index] to one-hot vectors as the paper suggested
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=10).float()
        y_index_one_hot = torch.nn.functional.one_hot(y[index], num_classes=10).float()
        mixed_y = lam * y_one_hot + (1 - lam) * y_index_one_hot

        return mixed_x, mixed_y