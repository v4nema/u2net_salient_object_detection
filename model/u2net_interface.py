import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB

from time import time

from config import path


class U2NetInterface:

    def __init__(self, selected_model='u2netp', device='cpu'):
        print('U2Net init')
        self.model_name: str = selected_model
        self.device: str = device
        self.model_dir = os.path.join(path.models, self.model_name, self.model_name + '.pth')
        self.image_dir = path.upload
        self.prediction_dir = path.u2net_output

        self.net = U2NET(3, 1) if self.model_name == 'u2net' else U2NETP(3, 1)
        if self.device != 'cpu':
            self.net.load_state_dict(torch.load(self.model_dir))
        else:
            self.net.load_state_dict(torch.load(self.model_dir, map_location=torch.device("cpu")))
        if torch.cuda.is_available():
            self.net.cuda()
        self.net.eval()

    # normalize the predicted SOD probability map
    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d - mi) / (ma - mi)

        return dn

    def save_output(self, image_name, pred, d_dir):
        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        im = Image.fromarray(predict_np * 255).convert('RGB')
        img_name = image_name.split(os.sep)[-1]
        image = io.imread(image_name)
        imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

        # pb_np = np.array(imo)

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        imo.save(d_dir + imidx + '.jpg')

    def predict(self, filename):
        img_name_list = [filename]

        # dataloader
        salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                       lbl_name_list=[],
                                       transform=transforms.Compose([RescaleT(320),
                                                                     ToTensorLab(flag=0)])
                                       )
        salobj_dataloader = DataLoader(salobj_dataset,
                                       batch_size=1,
                                       shuffle=False,
                                       num_workers=1)

        # print(len(salobj_dataloader))
        for i, data in enumerate(salobj_dataloader):

            print("inferencing:", img_name_list[i].split(os.sep)[-1])

            inputs_test = data['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1, d2, d3, d4, d5, d6, d7 = self.net(inputs_test)

            # normalization
            pred = d1[:, 0, :, :]
            pred = self.normPRED(pred)

            # save results to test_results folder
            if not os.path.exists(self.prediction_dir):
                os.makedirs(self.prediction_dir, exist_ok=True)
            self.save_output(img_name_list[i], pred, self.prediction_dir)

            del d1, d2, d3, d4, d5, d6, d7
