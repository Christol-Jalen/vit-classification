import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import random
from task import CustomDataset, polynomial_fun, gen_train_and_test_sets



def fit_polynomial_ls_sgd_learnable(x, t, lr, batch_size):
    """
    Fits a polynomial function to the given data points using minibatch gradient descent algorithm.
    It also learns the best M for polynomial fitting. The optimal M is selected so that it produce the least
    root mean square error between the predicted value and the ground-truth.

    Args:
    x (torch.Tensor: num of data points by 1): Input data points.
    t (torch.Tensor: num of data points by 1): Target values.
    lr (float): learning rate
    batch_size (int)

    Returns:
    best_model (torch.nn.Module): Best fitted polynomial model based on the lowest average loss.
    best_M (int): Degree of the best polynomial model.
    """

    print("Learning M...")
    print("-"*50)
    print("Principle: the M that generate least RMSEs between prediction and ground-truth is considered the optimal M")
    print("-"*50)

    M = 0
    least_rmse = float('inf')
    best_w = None
    best_M = None

    while True:
        M += 1
        print("trying M = " + str(M))
        # Generating polynomial features
        x_poly = torch.pow(x, torch.arange(M + 1, dtype=torch.float32))
        # Scaling input features
        x_poly_maxEachRow = torch.max(torch.abs(x_poly), dim=0).values
        x_poly_scaled = torch.div(x_poly,x_poly_maxEachRow)

        # Creating custom dataset
        dataset = CustomDataset(x_poly_scaled, t)
        # Initialize the data loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Creat linear model
        model = nn.Linear(M + 1, 1, bias=True, dtype=torch.float32)
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(model.weight)
        # Define SGD optimiser
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
        # Define loss function
        criterion = nn.MSELoss()
        # Training
        num_epochs = 15000
        for epoch in range(num_epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            if epoch % 1000 == 999:
                print("Epoch: " +  str(epoch+1) + " Loss: "+ str(loss.item()))
            
        # Calculate the rmse between the prediction and ground truth
        current_w = model.weight.reshape(M+1, 1)
        current_y_hat = polynomial_fun(current_w, x_poly_scaled)
        current_rmse = torch.sqrt(torch.mean((current_y_hat - t)**2))
        print("current root mean square error is: " + str(current_rmse.item()))
        # The least rmse indicate the best fitted model
        if current_rmse < least_rmse:
            least_rmse = current_rmse
            best_w = current_w
            best_M = M
        else:
            break

    print(f"Optimal polynomial degree: {best_M}, with root mean square error: {least_rmse}")

    return best_w, best_M






if __name__=="__main__":

    # parameters for generating dataset
    weight = torch.tensor([[2], [3], [4]], dtype=torch.float32)
    # observed training data; observed test data; true training data; true test data
    y_train, y_test, t_train, t_test = gen_train_and_test_sets(weight, 20, 10, 0.5)

    learning_rate = 0.03
    batch_size = 10

    # M is learnable
    w_hat_train, M = fit_polynomial_ls_sgd_learnable(y_train, t_train, lr=learning_rate, batch_size=batch_size)
    y_hat_train = polynomial_fun(w_hat_train, y_train)
    y_hat_test = polynomial_fun(w_hat_train, y_test)

    '''
    Report the optimised M value and the mean (and the standard deviation) in difference between the model-
    predicted values and the underlying "true" curve.
    '''
    print("\nThe optimal M value is: \n" + str(M))

    mean_diff_sgd = torch.mean(y_hat_train - t_train)
    std_diff_sgd =  (y_hat_train - t_train).std()
    print("\nThe mean difference between the model-predicted data and the true data is: \n" + str(mean_diff_sgd.item()))
    print("\nThe difference standard deviation between the model-predicted data and the true data is: \n" + str(std_diff_sgd.item()))