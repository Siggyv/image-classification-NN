import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    if training:
        dataset = datasets.FashionMNIST('./data',train=True, download=True, transform=custom_transform)
    else:
        dataset = datasets.FashionMNIST('./data', train=False, transform=custom_transform)
    
    return torch.utils.data.DataLoader(dataset, batch_size = 64)



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    #create optimizer
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    #two for loops
    for epoch in range(T):
        running_loss = 0.0
        correct = 0.0
        total = 0
        for i, data in enumerate(train_loader,0):
            inputs, labels = data

            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item() * labels.size(0)
            if i % 937 == 936:
                print(f"Train Epoch: {epoch}\t Accuracy: {correct}/{total} ({100 * correct / total:.2f}%) Loss: {running_loss / total:.3f}")

    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    #create optimizer
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    with torch.no_grad():
        numCorrect = 0
        total = 0
        total_loss = 0
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            numCorrect += (predicted==labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            
        if show_loss:
            print(f"Average Loss: {'{:.4f}'.format(round((total_loss / total), 4))}")
        print(f"Accuracy: {'{:.2f}'.format(round((100 * numCorrect / total), 2))}%")



def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    logs = model(test_images[index])
    prob = F.softmax(logs, dim = 1)
    idxval1 = (0,0)
    idxval2 = (0,0)
    idxval3 = (0,0)

    for idx,val in enumerate(prob[0]):
        val = val.item()
        if val > idxval1[1]:
            idxval1 = (idx, val)
        elif val > idxval2[1]:
            idxval2 = (idx,val)
        elif val > idxval3[1]:
            idxval3 = (idx,val)
    print(f"{class_names[idxval1[0]]}: {idxval1[1] * 100:.2f}%")
    print(f"{class_names[idxval2[0]]}: {idxval2[1] * 100:.2f}%")
    print(f"{class_names[idxval3[0]]}: {idxval3[1] * 100:.2f}%")    

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    model = build_model()
    model.train()
    criterion = nn.CrossEntropyLoss()
    epoch = 5
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    train_model(model, train_loader, criterion, epoch)
    model.eval()
    evaluate_model(model, test_loader, criterion, show_loss = True)
    tensor = torch.tensor([])
    for data in test_loader:
        images, label = data
        new_tensor = torch.tensor(images)
        tensor = torch.cat((tensor, new_tensor), dim=0)
    predict_label(model, tensor, 1)
    
