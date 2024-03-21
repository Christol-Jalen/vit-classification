import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import numpy as np

from network_pt import Net
from mixup_pt import mixup


def evaluate_model(net, dataloader):
    """
    This function evaluates the accuracy of the output of a network.

    Args:
        net: network.
        dataloader: test dataloader

    Returns:
        accuracy: the accuracy of the prediction

    """
    net.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) # 
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':

    # Apply cuda for acceleration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 16 # adjusted for mixup algorithm
    # Load training set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Load test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # example images
    dataiter = iter(trainloader)
    images, labels = next(dataiter) # note: for pytorch versions (<1.14) use dataiter.next()

    # performing mixup algorithm
    alpha = 0.3 # the paper suggested that alpha in [0.1, 0.4] will improve the model performance
    sampling_method = 1 # can choose sampling method, either 1 or 2
    MixUp = mixup(alpha, sampling_method)
    images, _ = MixUp.mixup_data(images, labels)

    im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("mixup.png") # show a batch of images
    print('mixup.png saved.') 
    print('Ground truth labels:' + ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    
    train_epochs = 20

    #=====================================Method 1=========================================
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
    for epoch in range(train_epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
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

            if i % 1000 == 999:   # print every 500 mini epoch
                 print('[epoch: %d, miniepoch: %5d] loss: %.3f' %(epoch + 1, i + 1, loss.item()))
        accuracy = evaluate_model(net_1, testloader)
        print("-"*30)
        print(f'Test Accuracy after epoch {epoch + 1}: {accuracy:.2f}%')
        print("-"*30)

    print('Method 1 training done.')
    print("="*30)

    # save trained model
    torch.save(net_1.state_dict(), 'saved_model_method1.pt')
    print('Model for method 1 is saved.')
    print("="*30)

    #=====================================Method 2===============================================
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
    for epoch in range(train_epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
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

            if i % 1000 == 999: # print every 500 mini epoch
                 print('[epoch: %d, miniepoch: %5d] loss: %.3f' %(epoch + 1, i + 1, loss.item()))
        accuracy = evaluate_model(net_2, testloader)
        print(f'Test Accuracy after epoch {epoch + 1}: {accuracy:.2f}%')

    print('Method 2 training done.')

    # save trained model
    torch.save(net_2.state_dict(), 'saved_model_method2.pt')
    print('Model for method 2 is saved.')
    print("="*30)


    #==========================================Result Visualising=============================================

    test_size = 36
    trainloader_result = torch.utils.data.DataLoader(trainset, batch_size=test_size, shuffle=True, num_workers=2)
    dataiter_result = iter(trainloader_result)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # inference
    images_result, labels_result = next(dataiter_result)
    # save to images
    im = Image.fromarray((torch.cat(images_result.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("result.png")

    images_result = images_result.to(device)
    labels_result = labels_result.to(device)
    print('ground-truth:\n', ' '.join('%5s' % classes[labels_result[j]] for j in range(test_size)))

    outputs_net1 = net_1(images_result)
    predicted = torch.argmax(outputs_net1, 1, keepdim=False)
    print('Method 1 model prediction:\n', ' '.join('%5s' % classes[predicted[j]] for j in range(test_size)))

    outputs_net2 = net_2(images_result)
    predicted = torch.argmax(outputs_net2, 1, keepdim=False)
    print('Method 2 model prediction:\n', ' '.join('%5s' % classes[predicted[j]] for j in range(test_size)))
