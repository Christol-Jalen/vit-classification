import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import numpy as np

from network_pt import Net
from mixup_pt import mixup

from torch.utils.data import random_split
import torch.nn.functional as F
import time
import math


'''
Validation Metric 1: Root Mean Square Error
'''
def calculate_rmse(loader, model, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_count = 0
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.MSELoss(reduction='sum') 
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            #inputs, labels = MixUp.mixup_data(inputs, labels) 
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_count += inputs.size(0)
    rmse = math.sqrt(total_loss / total_count)
    return rmse

'''
Validation Metric 2: F1 Score
'''
def calculate_f1_score(loader, model, device):
    model.eval()
    n_classes = 10  # CIFAR-10 has 10 classes
    class_correct = list(0. for _ in range(n_classes))
    class_total = list(0. for _ in range(n_classes))
    eps = 1e-9  # to avoid division by zero

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            #inputs, labels = MixUp.mixup_data(inputs, labels)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Calculating precision, recall, and F1 for each class
    precision = [class_correct[i] / (np.sum([predicted == i for predicted in class_correct]) + eps) for i in range(n_classes)]
    recall = [class_correct[i] / (class_total[i] + eps) for i in range(n_classes)]
    f1_scores = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + eps) for i in range(n_classes)]

    # Calculate average F1 score across all classes
    avg_f1_score = np.mean(f1_scores)
    return avg_f1_score

'''
Report a summary of loss values and the metrics on the holdout test set. Compare the results with those
obtained during development.
'''
def evaluate_on_holdout_test(loader, model, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    average_loss = 0.0
    total_count = 0
    n_classes = 10  # CIFAR-10 has 10 classes
    class_correct = list(0. for _ in range(n_classes))
    class_total = list(0. for _ in range(n_classes))
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.MSELoss(reduction='sum') 
    eps = 1e-9  # To avoid division by zero

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            #inputs, labels = MixUp.mixup_data(inputs, labels)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_count += inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Calculate Average Loss
    average_loss = total_loss / total_count        
    # Calculate RMSE
    rmse = math.sqrt(average_loss)

    # Calculate F1 Score
    precision = [class_correct[i] / (np.sum([predicted == i for predicted in class_correct]) + eps) for i in range(n_classes)]
    recall = [class_correct[i] / (class_total[i] + eps) for i in range(n_classes)]
    f1_scores = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + eps) for i in range(n_classes)]
    avg_f1_score = np.mean(f1_scores)

    return average_loss, rmse, avg_f1_score



if __name__ == '__main__':

    # Apply cuda for acceleration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 10 # adjusted for mixup algorithm
    # Load training set
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    '''
    Random split the data into development set (80%) and holdout test set (20%)
    
    '''
    # Calculate the sizes for 80/20 split
    total_size = len(dataset)
    print(f"Total  dataset size: {total_size}")
    development_size = int(total_size * 0.8)
    holdout_test_size = total_size - development_size
    development_set, holdout_test_set = random_split(dataset, [development_size, holdout_test_size])

    print(f"Development set size: {len(development_set)}")
    print(f"Holdout test set size: {len(holdout_test_set)}")

    '''
    Random split the development set into train set (90%) and validation set (10%)
    '''
    train_size = int(development_size * 0.9)
    validation_size = development_size - train_size
    train_set, validation_set = random_split(development_set, [train_size, validation_size])
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(validation_set)}")


    # Define data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=2)
    holdout_test_loader = torch.utils.data.DataLoader(holdout_test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # performing mixup algorithm
    alpha = 0.3 # the paper suggested that alpha in [0.1, 0.4] will improve the model performance
    sampling_method = 1 # can choose sampling method, either 1 or 2
    MixUp = mixup(alpha, sampling_method)
    
    train_epochs = 20

    #====================================Method 1==========================================
    ## VisionTransformer
    net_1 = Net(num_classes=len(classes)) # create a network for method 1
    net_1.to(device)

    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(net_1.parameters(), lr=0.001, momentum=0.9)

    sampling_method = 1
    print("="*30)
    print("Starting training the network using method 1")
    loss_net1_list = []
    rmse_net1_list = []
    f1_net1_list = []
    for epoch in range(train_epochs):  # loop over the dataset multiple times
        start_time = time.time()  # Start time of the epoch
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Apply mixup augmentation
            mixed_inputs, mixed_labels = MixUp.mixup_data(inputs, labels)

            mixed_inputs = mixed_inputs.to(device)
            mixed_labels = mixed_labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net_1(mixed_inputs)
            loss = criterion(outputs, mixed_labels) # MSE loss
            loss.backward()
            optimizer.step()

            if i % 500 == 499:   # print every 500 mini epoch
                 print('[epoch: %d, miniepoch: %5d] loss: %.3f' %(epoch + 1, i + 1, loss.item()))

        # Validation phase
        print("Epoch "+ str(epoch+1) + " Loss: " + str(loss.item()))       
        
        rmse = calculate_rmse(validation_loader, net_1, device)
        print("Epoch "+ str(epoch+1) + " Root Mean Square Error: " + str(rmse))

        avg_f1_score = calculate_f1_score(validation_loader, net_1, device)
        print("Epoch "+ str(epoch+1) + " F1 Score: " + str(rmse))

        end_time = time.time()  # End time of the epoch
        epoch_time = end_time - start_time  
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds.")

        loss_net1_list.append(loss.item())
        rmse_net1_list.append(rmse)
        f1_net1_list.append(avg_f1_score)
        

    print('Method 1 training done.')
    print("="*30)

    # save trained model
    torch.save(net_1.state_dict(), 'saved_model_method1.pt')
    print('Model for method 1 is saved.')
    print("="*30)

    #=====================================Method 2================================================
    ## VisionTransformer
    net_2 = Net(num_classes=len(classes)) # create a new network for method 2
    net_2.to(device)

    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(net_2.parameters(), lr=0.001, momentum=0.9)

    sampling_method = 2
    print("="*30)
    print("Starting training the network using method 2")
    loss_net2_list = []
    rmse_net2_list = []
    f1_net2_list = []
    for epoch in range(train_epochs):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Apply mixup augmentation
            mixed_inputs, mixed_labels = MixUp.mixup_data(inputs, labels)

            mixed_inputs = mixed_inputs.to(device)
            mixed_labels = mixed_labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net_2(mixed_inputs)
            loss = criterion(outputs, mixed_labels)
            loss.backward()
            optimizer.step()

            if i % 500 == 499: # print every 500 mini epoch
                 print('[epoch: %d, miniepoch: %5d] loss: %.3f' %(epoch + 1, i + 1, loss.item()))
                
        # Validation phase
        print("Epoch "+ str(epoch+1) + " Loss: " + str(loss.item()))

        rmse = calculate_rmse(validation_loader, net_2, device)
        print("Epoch "+ str(epoch+1) + " Root Mean Square Error: " + str(rmse))

        avg_f1_score = calculate_f1_score(validation_loader, net_2, device)
        print("Epoch "+ str(epoch+1) + " F1 Score: " + str(rmse))

        loss_net2_list.append(loss.item())
        rmse_net2_list.append(rmse)
        f1_net2_list.append(avg_f1_score)

    print('Method 2 training done.')

    # save trained model
    torch.save(net_2.state_dict(), 'saved_model_method2.pt')
    print('Model for method 2 is saved.')
    print("="*30)

    # Summary of loss values and metrics on the holdout test set
    loss_1, rmse_1, f1_1 = evaluate_on_holdout_test(holdout_test_loader, net_1, device)
    print(f"Method 1 on Holdout Test Set - Loss: {loss_1:.3f}, RMSE: {rmse_1:.3f}, F1 Score: {f1_1:.3f}")

    loss_2, rmse_2, f1_2 = evaluate_on_holdout_test(holdout_test_loader, net_2, device)
    print(f"Method 2 on Holdout Test Set - Loss: {loss_2:.3f}, RMSE: {rmse_2:.3f}, F1 Score: {f1_2:.3f}")

    # Compare the results on holdout test set to the results obtained during the development
    print("Method 1 Loss during development in epoch 1-20:")
    print(loss_net1_list)
    print("Method 1 RMSE during development in epoch 1-20:")
    print(rmse_net1_list)
    print("Method 1 F1 score during development in epoch 1-20:")
    print(f1_net1_list)

    print("Method 2 Loss during development in epoch 1-20:")
    print(loss_net2_list)
    print("Method 2 RMSE during development in epoch 1-20:")
    print(rmse_net2_list)
    print("Method 2 F1 score during development in epoch 1-20:")
    print(f1_net2_list)

    
   