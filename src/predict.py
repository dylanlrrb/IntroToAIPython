import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import json
import math
from PIL import Image
from train import build_network

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', help="path to checkpoint", default='./checkpoint.pt')
parser.add_argument('-i', '--image', help="image path", default='./flowers/test/85/image_04805.jpg')
parser.add_argument('-l', '--lables', help="JSON category mappings", default=None)
parser.add_argument('-g', '--gpu', type=bool, default=True)
parser.add_argument('-k', '--topk', type=int, default=5)
args = parser.parse_args()

if args.gpu == True:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

def load_model(path):
    checkpoint = torch.load(path, map_location={'cuda:0': str(device)})

    arch = checkpoint['arch']
    out_features = len(checkpoint['class_to_idx'])
    hidden_layers = checkpoint['hidden_layers']

    # For loading, use the same build_network function used to make saved model's network
    model = build_network(arch, out_features, hidden_layers)
    model.load_state_dict(checkpoint['state_dict'])

    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
    
loaded_model = load_model(args.checkpoint)
print("loaded checkpoint")

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open the image
    pil_image = Image.open(image_path)
    
    # Resize the image
    if pil_image.size[1] < pil_image.size[0]:
        pil_image.thumbnail((255, math.pow(255, 2)))
    else:
        pil_image.thumbnail((math.pow(255, 2), 255))
                            
    # Crop the image
    left = (pil_image.width-224)/2
    bottom = (pil_image.height-224)/2
    right = left + 224
    top = bottom + 224
                            
    pil_image = pil_image.crop((left, bottom, right, top))
                            
    # Turn into np_array
    np_image = np.array(pil_image)/255
    
    # Undo mean and std then transpose
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])  
    np_image = (np_image - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if topk < 1:
        topk = 1

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
    if str(device) == "cuda:0":
        probs = torch.exp(result[0].data).cpu().numpy()[0]
        idxs = result[1].data.cpu().numpy()[0]
    else:
        probs = torch.exp(result[0].data).numpy()[0]
        idxs = result[1].data.numpy()[0]
    
    return(probs, idxs)

test_image_path = args.image

print('using device:', device)
# Get the probabilties and indices from passing the image through the model
probs, idxs = predict(test_image_path, loaded_model, topk=args.topk)

# Swap the keys and values in class_to_idx so that
# indices can be mapped to original classes in dataset
idx_to_class = {v: k for k, v in loaded_model.class_to_idx.items()}

if args.lables != None:
    with open(args.lables, 'r') as f:
        cat_to_name = json.load(f)

    # Map the classes to flower category lables                              
    idxs = list(map(lambda x: cat_to_name[f"{idx_to_class[x]}"], idxs))
else:
    idxs = list(map(lambda x: idx_to_class[x], idxs))

topk = list(zip(idxs, probs))

print("\nMost Probable Class:")
print("{}: {:.1f}%".format(topk[0][0], topk[0][1] * 100))

print(f"\n---Top {args.topk} predictions---")

for i in topk:
    print("{}: {:.1f}%".format(i[0], i[1] * 100))
    
