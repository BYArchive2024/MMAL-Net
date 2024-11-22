import torch
import torch.nn as nn
import sys
from config import input_size, proposalN, channels
from utils.read_dataset import read_dataset
from utils.auto_laod_resume import auto_load_resume
from networks.model import MainNet
from torchvision import transforms
import json
import imageio
import numpy as np
from PIL import Image
import pandas as pd

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CUDA = torch.cuda.is_available()
# DEVICE = torch.device("cuda" if CUDA else "cpu")
DEVICE = torch.device("cpu")

root = './datasets/FGVC-aircraft'  # dataset path
# model path
pth_path = "./models/air_epoch146.pth"
num_classes = 100

batch_size = 10

_, testloader = read_dataset(input_size, batch_size, root, "Aircraft")

model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()

if os.path.exists(pth_path):
    epoch = auto_load_resume(model, pth_path, status='test')
else:
    sys.exit('There is not a pth exist.')

with open('id_label.json', 'r') as f:
    id_label_data = json.load(f)

id_label = {int(k): v for k, v in id_label_data.items()}

def preprocess_image(img):
    if len(img.shape) == 2:
        img = np.stack([img] * 3, 2)
    img = Image.fromarray(img, mode='RGB')
    img = transforms.Resize((448, 448), Image.BILINEAR)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    img = img.unsqueeze(0)
    return img

model.eval()

def predict(path: str):
    img = preprocess_image(imageio.v3.imread(path))

    with torch.no_grad():
        output = model.forward(img, None, None, "test", "cpu")[-2:]
        pred = output[0].max(1, keepdim=True)[1]
        
        result = pred.numpy()[0][0]
        result_label = id_label[result]

        return result_label

df = pd.read_excel("info.xlsx")

def get_info(name: str):
    return df[df.기종 == name]

r = predict("image.png")
info = get_info(r)

print(info)