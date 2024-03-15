import numpy as np
import torch

'''
Implement a polynomial function polynomial_fun, that takes two input arguments, a weight vector 𝐰
of size 𝑀 + 1 and an input scalar variable 𝑥, and returns the function value 𝑦. The polynomial_fun
should be vectorised for multiple pairs of scalar input and output, with the same 𝐰. [5]
'''
def polynomial_fun(w, x): # checked
    """
    Compute the polynomial function value y for input scalar variable x
    with weight vector w.

    Args:
    - w (numpy array: num of data points by 1): weight vector
    - x (numpy array: num of data points by 1): input vector

    Returns:
    - y (numpy array): array of function values
    """
    # # Ensure w is a numpy array
    w = np.array(w)
    x = np.array(x)
    
    size = w.shape[0]
    powers_of_x = np.power(x, np.arange(size))
    y = np.matmul(powers_of_x, w)
    
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
    x_poly = torch.pow(x, torch.arange(M+1, dtype=torch.float32))
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
    t_train (torch.Tensor: numTrainSamples by 1): training set
    t_test (torch.Tensor: numTestSamples by 1): test set
    """

    x_train = 40.0*(torch.rand(numTrainSamples, dtype=torch.float32) - 0.5).reshape(20,1)
    y_train = torch.tensor(polynomial_fun(w, x_train), dtype=torch.float32) # converte to torch tensor
    noise_train = torch.randn(numTrainSamples).reshape(20,1) * noiseStdDev
    t_train = y_train + noise_train

    x_test = 40.0*(torch.rand(numTestSamples, dtype=torch.float32) - 0.5).reshape(10,1)
    y_test = torch.tensor(polynomial_fun(w, x_test), dtype=torch.float32) # converte to torch tensor
    noise_test = torch.randn(numTestSamples).reshape(10,1) * noiseStdDev
    t_test = y_test + noise_test

    return y_train, y_test, t_train, t_test


def main():

    '''
    Use polynomial_fun (𝑀 = 2, 𝐰 = [1,2,3]^T) to generate a training set and a test set, in the
    form of respectively and uniformly sampled 20 and 10 pairs of 𝑥, 𝑥𝜖[−20, 20], and 𝑡. The 
    observed 𝑡 values are obtained by adding Gaussian noise (standard deviation being 0.5) to 𝑦.
    '''

    # generate data sets with M = 2 and w = [1,2,3]^T
    y_train, y_test, t_train, t_test = gen_train_and_test_sets(2, [[1], [2], [3]], 20, 10, 0.5)

    # print("%s" % "the training set is:")
    # print(t_train)
    
    # print("%s" % "the test set is:")
    # print(t_test)

    '''
    Use fit_polynomial_ls (𝑀𝜖{2,3,4}) to compute the optimum weight vector 𝐰̂ using the
    training set. For each 𝑀, compute the predicted target values 𝑦̂ for all 𝑥 in both the training
    and test sets.
    '''

    # M = 2
    w_hat_M2_train = fit_polynomial_ls(y_train, t_train, M=2)
    y_hat_M2_train = polynomial_fun(w_hat_M2_train, y_train)
    print(w_hat_M2_train)
    print(y_hat_M2_train)

    w_hat_M2_test = fit_polynomial_ls(y_test, t_test, M=2)
    y_hat_M2_test = polynomial_fun(w_hat_M2_test, y_test)
    print(w_hat_M2_test)
    print(y_hat_M2_test)

    # M = 3
    w_hat_M3_train = fit_polynomial_ls(y_train, t_train, M=3)
    y_hat_M3_train = polynomial_fun(w_hat_M3_train, y_train)
    print(w_hat_M3_train)
    print(y_hat_M3_train)

    w_hat_M3_test = fit_polynomial_ls(y_test, t_test, M=2)
    y_hat_M3_test = polynomial_fun(w_hat_M3_test, y_test)
    print(w_hat_M3_test)
    print(y_hat_M3_test)

    # M = 4
    w_hat_M4_train = fit_polynomial_ls(y_train, t_train, M=4)
    y_hat_M4_train = polynomial_fun(w_hat_M4_train, y_train)
    print(w_hat_M4_train)
    print(y_hat_M4_train)

    w_hat_M4_test = fit_polynomial_ls(y_test, t_test, M=4)
    y_hat_M4_test = polynomial_fun(w_hat_M4_test, y_test)
    print(w_hat_M4_test)
    print(y_hat_M4_test)




    return None


if __name__=="__main__":
    main()