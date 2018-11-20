import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse

def build_network(archit="vgg16", out_features=102, hidden_layers=[1000]):
    model = getattr(models, archit)(pretrained=True)

    # Freeze the params from training
    for param in model.parameters():
        param.requires_grad = False

    # Find the number of in features the classifier expects
    try:
        iter(model.classifier)
    except TypeError:
        in_features = model.classifier.in_features
    else:
        in_features = model.classifier[0].in_features
    
    hidden_layers = [in_features] + hidden_layers
    
    # Define how to build a hidden layer
    layer_builder = (
        lambda i, v: (f"fc{i}", nn.Linear(hidden_layers[i-1], v)),
        lambda i, v: (f"relu{i}", nn.ReLU()),
        lambda i, v: (f"drop{i}", nn.Dropout())
    )
    
    # Loop through hidden_layers array and build a layer for each
    layers = [f(i, v) for i, v in enumerate(hidden_layers) if i > 0 for f in layer_builder]
    # Concat the output layer onto the network
    layers += [('fc_final', nn.Linear(hidden_layers[-1], out_features)),
               ('output', nn.LogSoftmax(dim=1))]
    
    classifier = nn.Sequential(OrderedDict(layers))

    # Replace classifier with our classifier
    model.classifier = classifier

    return model



def validate(model, criterion, testing_set, device):
    accuracy = 0
    test_loss = 0
    model.eval() # Evaluation mode
    with torch.no_grad():
        for images, labels in testing_set:
            
            images = images.to(device)
            labels = labels.to(device)

            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            # Take exponential to get log softmax probibilities
            probs = torch.exp(output)

            # highest probability is the predicted class
            # compare with true label
            correct_predictions = (labels.data == probs.max(1)[1])

            # Turn ByteTensor into np_array to calculate mean
            accuracy += np.array(correct_predictions).mean()
    
    model.train() # Switch training mode back on
    
    return test_loss/len(testing_set), accuracy/len(testing_set)

def train(model, criterion, optimizer, training_set, testing_set, device, epochs=3, log_frequency=20):
    
    batch = 0
    train_loss = 0
    for epoch in range(epochs):
        model.train() # Turn on training mode
        
        for images, labels in training_set:

            batch += 1
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # Prevent gradients from accumulating
            
            output = model.forward(images)
            loss = criterion(output, labels) # Calculate loss
            # Back-prop
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if batch % log_frequency == 0:
                
                test_loss, accuracy = validate(model, criterion, testing_set, device)
                
                print("Epoch: {} of {}, ".format(epoch+1, epochs),
                      "Train Loss: {:.3f}, ".format(train_loss/log_frequency),
                      "Test Loss: {:.3f}, ".format(test_loss),
                      "Accuracy: %{:.1f}".format(accuracy*100))
                
                train_loss = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', help="vgg or densenet", default='vgg')
    parser.add_argument('-d', '--data_dir', default='flowers')
    parser.add_argument('-hl', '--hidden_layers', type=int, default=1000)
    parser.add_argument('-e', '--epochs', type=int, default=3)
    parser.add_argument('-g', '--gpu', type=bool, default=True)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    if args.arch == 'vgg':
        arch = 'vgg16'
    elif args.arch == 'densenet':
        arch = 'densenet121'
    else:
        raise ValueError('Unexpected architecture', args.arch)

    if args.gpu == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define normalization transform for reuse
    noramlize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    noramlize]),
        'valid': transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    noramlize]),
        'test': transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    noramlize])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test':  datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # define the dataloaders using the image datasets and the trainforms,
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True),
        'test':  torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True)
    }

    model = build_network(archit=arch,
                          out_features=len(image_datasets['train'].class_to_idx),
                          hidden_layers=[args.hidden_layers])

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    print("using ", device)
    model.to(device)
    print("begin training")
    train(model, criterion, optimizer, dataloaders['train'], dataloaders['valid'], device, epochs=args.epochs, log_frequency=20)
    print("trained\n\n")

    # Check the test loss and accuracy of the trained network
    test_loss, accuracy = validate(model, criterion, dataloaders['test'], device)
    print("Network preformance on test dataset-------------")
    print("Accuracy on test dataset: %{:.1f}".format(accuracy*100))

    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {
        "arch": arch,
        "class_to_idx": model.class_to_idx,
        'state_dict': model.state_dict(),
        "hidden_layers": [args.hidden_layers]
    }

    torch.save(checkpoint, "checkpoint.pt")
    print("saved checkpoint")

if __name__ == "__main__":
    main()
