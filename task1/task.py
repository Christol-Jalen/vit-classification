import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import random
import time

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        """
        Initialize the dataset with input data and target data.

        Args:
        x_data (torch.Tensor or numpy.ndarray): Input data.
        y_data (torch.Tensor or numpy.ndarray): Target data.
        """
        self.x_data = torch.tensor(x_data, dtype=torch.float32) if not torch.is_tensor(x_data) else x_data
        self.y_data = torch.tensor(y_data, dtype=torch.float32) if not torch.is_tensor(y_data) else y_data

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset at the given index.

        Args:
        idx (int): Index of the sample to retrieve.

        Returns:
        tuple: Tuple containing the input data and target data for the sample.
        """
        x_sample = self.x_data[idx]
        y_sample = self.y_data[idx]
        return x_sample, y_sample

'''
Implement a polynomial function polynomial_fun, that takes two input arguments, a weight vector 𝐰
of size 𝑀 + 1 and an input scalar variable 𝑥, and returns the function value 𝑦. The polynomial_fun
should be vectorised for multiple pairs of scalar input and output, with the same 𝐰. [5]
'''
def polynomial_fun(w, x):
    """
    Compute the polynomial function value y for input scalar variable x
    with weight vector w.

    Args:
    - w (torch.Tensor: num of data points by 1): weight vector
    - x (torch.Tensor: num of data points by 1): input vector

    Returns:
    - y (torch.Tensor): tensor of function values
    """
    # Ensure w and x are PyTorch tensors
    w = torch.tensor(w, dtype=torch.float32)
    x = torch.tensor(x, dtype=torch.float32)
    
    size = w.shape[0]
    powers_of_x = torch.pow(x, torch.arange(size, dtype=torch.float32))
    y = torch.matmul(powers_of_x, w)
    
    return y

# # Testing:
# w = [[1], [2], [3]]  # Weight vector
# x = [[5], [6]]  # Input scalar variable
# y = polynomial_fun(w, x)
# print(y) 

'''
Using the linear algebra modules in TensorFlow/PyTorch, implement a least square solver for fitting
the polynomial functions, fit_polynomial_ls, which takes 𝑁 pairs of 𝑥 and target values 𝑡 as input, with
an additional input argument to specify the polynomial degree 𝑀, and returns the optimum weight
vector 𝐰̂ in least-square sense, i.e. ‖𝑡 − 𝑦‖^2 is minimised. [5]
'''
def fit_polynomial_ls(x,t,M):
    """
    Fits a polynomial function to the given data points using least squares method.

    Args:
    x (torch.Tensor: num of data points by 1): Input data points.
    t (torch.Tensor: num of data points by 1): Target values.
    M (int): Degree of the polynomial.

    Returns:
    w_hat (torch.Tensor: num of data points by 1): Optimal weight vector.
    """
    # Generating polynomials
    x_poly = torch.pow(x, torch.arange(M+1, dtype=torch.float32))

    # Computing optimal weights
    w_hat = torch.linalg.lstsq(x_poly, t).solution

    return w_hat


# # Testing
# x = torch.tensor([[5], [6], [7], [8]], dtype=torch.float32)
# # Some polynomial function as target values
# t = 5 + 3*x  + 2*x**2
# # Fit a polynomial of degree 2
# M = 2
# w = fit_polynomial_ls(x, t, M)
# print("Optimal weight vector:", w)

def fit_polynomial_ls_sgd(x,t,M, lr, batch_size):
    """
    Fits a polynomial function to the given data points using minibatch gradient descent algorithm.

    Args:
    x (torch.Tensor: num of data points by 1): Input data points.
    t (torch.Tensor: num of data points by 1): Target values.
    M (int): Degree of the polynomial.
    lr (float): learning rate
    batch_size (int)

    Returns:
    w_hat (torch.Tensor: num of data points by 1): Optimal weight vector.
    """

    print("Starting gradient descent algorithm")

    # Generating polynomials
    x_poly = torch.pow(x, torch.arange(M+1, dtype=torch.float32))
    # Input Scalling
    x_poly_maxEachRow = torch.max(torch.abs(x_poly), dim=0).values
    x_poly_scaled = torch.div(x_poly,x_poly_maxEachRow)
    # Defining dataset
    customDataset = CustomDataset(x_poly_scaled, t)

    # Initialize the data loader
    dataloader = DataLoader(customDataset, batch_size=batch_size, shuffle=True)

    # Initialize weights
    w_hat = torch.zeros(M + 1, 1, dtype=torch.float32, requires_grad=True)
    
    # Create a Linear Model: prediction = w_hat * input
    input_dim = M+1
    output_dim = 1
    model = nn.Linear(input_dim, output_dim, bias=False, dtype=torch.float32)

    # Define loss function
    criterion = nn.MSELoss() # mean squared error
    # Define optimiser
    optimiser = torch.optim.SGD(model.parameters(), lr) # stochastic gradient descent optimiser
    
    # Tranining parameters
    num_epochs = 5000

    for epoch in range(num_epochs):
        for input, target in dataloader:
            # zero the gradient before each iteraction
            optimiser.zero_grad()
            # forward prpagation
            prediction = model(input) 
            # calculating loss between prediction and target
            loss = criterion(prediction, target) 
            # back propagation
            loss.backward()
            # start optimisation
            optimiser.step()

        if epoch % 1000 == 0:
            print("Epoch: " +  str(epoch) + " Loss: "+ str(loss.item()))
    
    print("=" * 50)

    w_hat = model.weight.reshape(M+1, 1)
    return w_hat



def gen_train_and_test_sets(M, w, numTrainSamples, numTestSamples, noiseStdDev):

    """
    Generate training set and test set

    Args:
    M (int): Degree of the polynomial.
    w (numpy array: num of data points by 1): weight vector
    numTrainSamples (int): number of training samples
    numTestSamples (int): number of test samples
    noiseStdDev (int): the standard deviation of noise added to the dataset
    
    Returns:
    y_train (torch.Tensor: numTrainSamples by 1): training set
    y_test (torch.Tensor: numTestSamples by 1): test set
    t_train (torch.Tensor: numTrainSamples by 1): training set - ground truth
    t_test (torch.Tensor: numTestSamples by 1): test set - ground truth
    """

    x_train = torch.tensor([random.uniform(-20, 20) for _ in range(numTrainSamples)]).reshape(numTrainSamples,1)
    y_train = torch.tensor(polynomial_fun(w, x_train), dtype=torch.float32) # converte to torch tensor
    noise_train = torch.randn(numTrainSamples).reshape(20,1) * noiseStdDev
    t_train = y_train + noise_train

    x_test = torch.tensor([random.uniform(-20, 20) for _ in range(numTestSamples)]).reshape(numTestSamples,1)
    y_test = torch.tensor(polynomial_fun(w, x_test), dtype=torch.float32) # converte to torch tensor
    noise_test = torch.randn(numTestSamples).reshape(10,1) * noiseStdDev
    t_test = y_test + noise_test

    return y_train, y_test, t_train, t_test


if __name__=="__main__":

    '''
    Use polynomial_fun (𝑀 = 2, 𝐰 = [1,2,3]^T) to generate a training set and a test set, in the
    form of respectively and uniformly sampled 20 and 10 pairs of 𝑥, 𝑥𝜖[−20, 20], and 𝑡. The 
    observed 𝑡 values are obtained by adding Gaussian noise (standard deviation being 0.5) to 𝑦.
    '''
    # observed training data; observed test data; true training data; true test data
    y_train, y_test, t_train, t_test = gen_train_and_test_sets(2, [[1], [2], [3]], 20, 10, 0.5)

    # print("%s" % "the test set is:")
    # print(t_test)

    '''
    Use fit_polynomial_ls (𝑀𝜖{2,3,4}) to compute the optimum weight vector 𝐰̂ using the
    training set. For each 𝑀, compute the predicted target values 𝑦̂ for all 𝑥 in both the training
    and test sets.
    '''

    print("="*50)
    print("LS prediction")
    
    '''
    a) Report the mean and standard deviation in the difference between observed training data and
    underlying "true" polynomial curve
    '''
    mean_diff_ob = torch.mean(y_train - t_train)
    std_diff_ob =  (y_train - t_train).std()
    print("The mean difference between the observed training data and the true data is: \n" + str(mean_diff_ob.item()))
    print("The standard deviation between observed training data and the true data is: \n" + str(std_diff_ob.item()))

    for M in [2,3,4]:
        print("-"*50)
        print("M = " + str(M))
        w_hat_train = fit_polynomial_ls(y_train, t_train, M=M) # predicted weight
        y_hat_train = polynomial_fun(w_hat_train, y_train) # LS predicted training data
        y_hat_test = polynomial_fun(w_hat_train, y_test) # LS predicted test data

        '''
        b) Report the mean and standard deviation in the difference between the "LS-predicted" values and
        underlying "true" polynomial curve
        '''
        mean_diff_ls = torch.mean(y_hat_train - t_train)
        std_diff_ls =  (y_hat_train - t_train).std()
        print("The mean difference between the LS-predicted training data and the true data is: \n" + str(mean_diff_ls.item()))
        print("The standard deviation between the LS-predicted training data and the true data is: \n" + str(std_diff_ls.item()))



    '''
    Use fit_polynomial_sgd (𝑀𝜖{2,3,4}) to optimise the weight vector 𝐰̂ using the training set. For each 𝑀, compute the predicted target values 𝑦̂ for all 𝑥 in both the training and test sets.
    ''' 
    print("="*50)
    print("SGD prediction")   
    learning_rate = 0.05
    batch_size = 10

    for M in [2,3,4]:
        print("-"*50)
        print("M = " + str(M))
        w_hat_train_sgd = fit_polynomial_ls_sgd(y_train, t_train, M=M, lr=learning_rate, batch_size=batch_size)
        y_hat_train_sgd = polynomial_fun(w_hat_train_sgd, y_train)
        y_hat_test_sgd = polynomial_fun(w_hat_train_sgd, y_test)

        '''
        Report the mean and standard deviation in the difference between the "SGD-predicted" values and
        underlying "true" polynomial curve
        '''
        mean_diff_sgd = torch.mean(y_hat_train_sgd - t_train)
        std_diff_sgd =  (y_hat_train_sgd - t_train).std()
        print("The mean difference between the SGD-predicted training data and the true data is: \n" + str(mean_diff_sgd.item()))
        print("The standard deviation between the SGD-predicted training data and the true data is: \n" + str(std_diff_sgd.item()))


