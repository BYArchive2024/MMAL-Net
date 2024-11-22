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
from PIL import Image, ImageTk
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

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

def predict_with_path(path: str):
    img = preprocess_image(imageio.v3.imread(path))

    with torch.no_grad():
        output = model.forward(img, None, None, "test", "cpu")[-2:]
        pred = output[0].max(1, keepdim=True)[1]
        
        result = pred.numpy()[0][0]
        result_label = id_label[result]

        return result_label

def predict_with_image(img):
    img = preprocess_image(img)

    with torch.no_grad():
        output = model.forward(img, None, None, "test", "cpu")[-2:]
        pred = output[0].max(1, keepdim=True)[1]
        
        result = pred.numpy()[0][0]
        result_label = id_label[result]

        return result_label

df = pd.read_excel("info.xlsx")

def get_info(name: str):
    return df[df.기종 == name]

class ImageInfoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("이미지 정보 추출기")
        self.root.geometry("800x500")
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.left_frame = tk.Frame(self.main_frame, width=400, height=400)
        self.left_frame.grid(row=0, column=0, padx=20, pady=20)
        self.left_frame.pack_propagate(False)
        self.right_frame = tk.Frame(self.main_frame, width=400, height=400)
        self.right_frame.grid(row=0, column=1, padx=20, pady=20)
        self.right_frame.pack_propagate(False)
        self.image_label = tk.Label(self.left_frame, text="이미지를 선택해주세요.", width=40, height=10)
        self.image_label.pack(padx=10, pady=10)
        self.select_button = tk.Button(self.right_frame, text="이미지 선택", command=self.select_image)
        self.select_button.pack(pady=10)
        self.info_text = tk.Text(self.right_frame, height=20, width=40)
        self.info_text.pack(padx=10, pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(title="이미지 선택", filetypes=(("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif", "*.webp"), ("All Files", "*.*")))
        if file_path:
            try:
                self.image = Image.open(file_path)
                image_info = self.get_image_info(self.image)
                self.show_image(file_path)
                self.show_info(image_info)
            except Exception as e:
                messagebox.showerror("오류", f"이미지 처리 중 오류가 발생했습니다: {e}")

    def show_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((350, 350))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk, text="")
        self.image_label.image = img_tk

    def get_image_info(self, image):
        img_array = np.array(image.convert("RGB"))
        result = predict_with_image(img_array)
        info = get_info(result)
        s = ""
        for i in info.columns:
            s += f"{i}: {info[i].values[0]}\n"
        return s

    def show_info(self, info):
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageInfoApp(root)
    root.mainloop()
