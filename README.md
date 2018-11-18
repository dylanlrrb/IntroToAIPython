
# Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

![Cover](./final_project/assets/Flowers.png)

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.


```python
%config IPCompleter.greedy=True
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
```

## Load the data

Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
 


```python
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
```


```python
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
```

### Label mapping

You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.


```python
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
```

# Building and training the classifier

Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.

We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students! You can also ask questions on the forums or join the instructors in office hours.

Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:

* Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
* Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
* Train the classifier layers using backpropagation using the pre-trained network to get the features
* Track the loss and accuracy on the validation set to determine the best hyperparameters

We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!

When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.


```python
# Load network as feature detector
def build_network(arch="vgg13", out_features=102, hidden_layers=[1000]):
    model = getattr(models, arch)(pretrained=True)

    # Freeze the params from training
    for param in model.parameters():
        param.requires_grad = False\

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
        lambda i, v: (f"drop{i}", nn.Dropout()), 
        lambda i, v: (f"fc{i}", nn.Linear(hidden_layers[i-1], v)),
        lambda i, v: (f"relu{i}", nn.ReLU()),
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

model = build_network()
print(model)
```

    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace)
        (14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (15): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (16): ReLU(inplace)
        (17): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace)
        (19): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (20): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (21): ReLU(inplace)
        (22): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (23): ReLU(inplace)
        (24): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (classifier): Sequential(
        (drop1): Dropout(p=0.5)
        (fc1): Linear(in_features=25088, out_features=1000, bias=True)
        (relu1): ReLU()
        (fc_final): Linear(in_features=1000, out_features=102, bias=True)
        (output): LogSoftmax()
      )
    )



```python
def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    model.eval() # Evaluation mode
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            # Take exponential to get log softmax probibilities
            ps = torch.exp(output)

            # highest probability is the predicted class
            # compare with true label
            equality = (labels.data == ps.max(1)[1])

            # Turn ByteTensor into np_array to calculate mean
            accuracy += np.array(equality).mean()
    
    model.train() # Switch training mode back on
    
    return test_loss, accuracy

def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40):
    
    steps = 0
    running_loss = 0
    for e in range(epochs):
        model.train() # Turn on training mode
        
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() # Prevent gradients from accumulating
            
            output = model.forward(images)
            loss = criterion(output, labels) # Calculate loss
            # Back-prop
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if steps % print_every == 0:
 
                model.eval()
                
                test_loss, accuracy = validation(model, testloader, criterion)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                running_loss = 0
                
                model.train()
```


```python
# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("training on device:", device)
model.to(device);
```

    training on device: cuda:0



```python
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

print("begin training")
train(model, dataloaders['train'], dataloaders['valid'], criterion, optimizer, epochs=3, print_every=20)
print("trained")
```

    begin training
    Epoch: 1/3..  Training Loss: 3.934..  Test Loss: 2.310..  Test Accuracy: 0.470
    Epoch: 1/3..  Training Loss: 2.326..  Test Loss: 1.312..  Test Accuracy: 0.648
    Epoch: 1/3..  Training Loss: 1.740..  Test Loss: 0.956..  Test Accuracy: 0.728
    Epoch: 1/3..  Training Loss: 1.431..  Test Loss: 0.746..  Test Accuracy: 0.795
    Epoch: 1/3..  Training Loss: 1.272..  Test Loss: 0.712..  Test Accuracy: 0.799
    Epoch: 2/3..  Training Loss: 1.186..  Test Loss: 0.643..  Test Accuracy: 0.819
    Epoch: 2/3..  Training Loss: 1.105..  Test Loss: 0.570..  Test Accuracy: 0.840
    Epoch: 2/3..  Training Loss: 1.064..  Test Loss: 0.623..  Test Accuracy: 0.827
    Epoch: 2/3..  Training Loss: 0.948..  Test Loss: 0.552..  Test Accuracy: 0.854
    Epoch: 2/3..  Training Loss: 1.085..  Test Loss: 0.574..  Test Accuracy: 0.841
    Epoch: 3/3..  Training Loss: 0.954..  Test Loss: 0.509..  Test Accuracy: 0.860
    Epoch: 3/3..  Training Loss: 0.925..  Test Loss: 0.461..  Test Accuracy: 0.861
    Epoch: 3/3..  Training Loss: 0.918..  Test Loss: 0.444..  Test Accuracy: 0.879
    Epoch: 3/3..  Training Loss: 0.859..  Test Loss: 0.473..  Test Accuracy: 0.875
    Epoch: 3/3..  Training Loss: 0.899..  Test Loss: 0.431..  Test Accuracy: 0.871
    trained


## Testing your network

It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.


```python
# Check the test loss and accuracy of the trained network
test_loss, accuracy = validation(model, dataloaders['test'], criterion)
print("Network preformance on test dataset-------------")
print("Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders['test'])),
                      "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders['test'])))
```

    Network preformance on test dataset-------------
    Test Loss: 0.497..  Test Accuracy: 0.875


## Save the checkpoint

Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.

```model.class_to_idx = image_datasets['train'].class_to_idx```

Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.


```python
model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {
    "arch": "vgg13",
    "class_to_idx": model.class_to_idx,
    'state_dict': model.state_dict(),
    "hidden_layers": [1000]
}

torch.save(checkpoint, "checkpoint.pt")
```

## Loading the checkpoint

At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.


```python
def load_model(path): # 'checkpoint.pt'
    checkpoint = torch.load(path)

    arch = checkpoint['arch']
    out_features = len(checkpoint['class_to_idx'])
    hidden_layers = checkpoint['hidden_layers']

    # For loading, use the same build_network function used to make saved model's network
    model = build_network(arch, out_features, hidden_layers)
    model.load_state_dict(checkpoint['state_dict'])

    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
    
loaded_model = load_model('checkpoint.pt')
```

# Inference for classification

Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```

First you'll need to handle processing the input image such that it can be used in your network. 

## Image Preprocessing

You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.

As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 

And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.


```python
from PIL import Image
import numpy as np

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Define same transforms that are used in training images
    image_loader = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor()])
    # open the image
    pil_image = Image.open(image_path)
    # transform the image
    pil_image = image_loader(pil_image)
    # Turn into np_array
    np_image = np.array(pil_image)
    
    # undo mean and std then transpose
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image
```

To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).


```python
def imshow(image, ax=None, title=''):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.set_title(title)
    
    plt.show(ax)
    
    return ax
```

## Class Prediction

Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.

To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```


```python
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.eval()
    
    # Turn the np_array image into a FloatTensor
    # before running through model
    tensor_img = torch.FloatTensor([process_image(image_path)])
    # Note: model expects an 1D array of tensors of size (2, 244, 244)
    # so that's why I'm putting the result of process_image()
    # in brackets before passing into FloatTensor()
    # in other words, model expects a tensor of size (1, 3, 244, 244)
    # you coud also use `tensor_img = tensor_img.unsqueeze(0)`
    
    tensor_img = tensor_img.to(device)
    
    result = model(tensor_img).topk(topk)
    
    # Take the natural exponent of each probablility
    # to undo the natural log from the NLLLoss criterion
    probs = torch.exp(result[0].data).cpu().numpy()[0]
    # .cpu() to take the tensor off gpu
    # so it can be turned into a np_array
    idxs = result[1].data.cpu().numpy()[0]
    
    return(probs, idxs)

```

## Sanity Checking

Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:

<img src='./final_project/assets/Flowers.pngassets/inference_example.png' width=300px>

You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.


```python
test_image_path = test_dir + '/85/image_04805.jpg'
test_image_name = cat_to_name[f"{test_image_path.split('/')[-2]}"]

# Show the image for reference
imshow(process_image(test_image_path), title=test_image_name);

# Get the probabilties and indices from passing the image through the model
probs, idxs = predict(test_image_path, loaded_model)

# Swap the keys and values in class_to_idx so that
# indices can be mapped to original classes in dataset
idx_to_class = {v: k for k, v in loaded_model.class_to_idx.items()}

# Map the classes to flower category lables                              
names = list(map(lambda x: cat_to_name[f"{idx_to_class[x]}"], idxs))

# Display top 5 most probable flower categories                               
y_pos = np.arange(len(names))
plt.barh(y_pos, probs, align='center')
plt.yticks(y_pos, names)
plt.gca().invert_yaxis()
 
plt.show()
```


![png](output_25_0.png)



![png](output_25_1.png)



```python

```
