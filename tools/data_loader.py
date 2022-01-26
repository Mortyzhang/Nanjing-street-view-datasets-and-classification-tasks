from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import torch
import cv2
import tools.transform_func as T
import argparse


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return dirs
        else:
            return file


class MakeList(object):
    """
    this class used to make list of data for model train and test, return the root name of each image
    root: txt file records condition for every cxr image
    """
    def __init__(self, args, ratio=0.8):
        self.image_root = args.dataset_dir
        self.csv_root = args.csv_root
        self.convert = {"A": 0, "B": 1, "C": 2, "D": 3}
        self.ration = ratio
        self.multi = args.multi_label

    def read_csv(self):
        all_data = []
        data = pd.read_csv(self.csv_root)
        data = data[["No", "Type"]].values
        for i in range(len(data)):
            label = self.deal_label(data[i][1])
            all_data.append([os.path.join(self.image_root, str(data[i][0])+".jpg"), label])
        train, val = train_test_split(all_data, random_state=1, train_size=self.ration)
        return train, val

    def deal_label(self, original_label):
        label = original_label.replace('"', "")
        label = label.split(",")
        if not self.multi:
            if len(label) > 1:
                back = 4 #几个分类
            else:
                back = self.convert[label[0]]
        else:
            back = np.zeros(4, dtype=int) #几个分类
            for i in range(len(label)):
                back[self.convert[label[i]]] = 1
        return back


class MakeListInference(object):
    """
    this class used to make list of data for model train and test, return the root name of each image
    root: txt file records condition for every cxr image
    """
    def __init__(self, args):
        self.image_root = args.inference_dir

    def load_folder_file(self):
        total = []
        file_name = sorted(get_name(self.image_root, mode_folder=False))
        for dir in file_name:
            total.append([self.image_root+dir, 0])
        return total


class CityDataset(Dataset):
    """read all image name and label"""
    def __init__(self, data, args, transform=None):
        self.all_item = data
        self.args = args
        self.transform = transform

    def __len__(self):
        return len(self.all_item)

    def __getitem__(self, item_id):  # generate data when giving index
        while not os.path.exists(self.all_item[item_id][0]):
            print("not exist image:" + self.all_item[item_id][0])
            break
        image_name = self.all_item[item_id][0]
        image = cv2.imread(image_name)   # cv2.IMREAD_GRAYSCALE
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        image = cv2.resize(image, (self.args.img_size, self.args.img_size), interpolation=cv2.INTER_LINEAR)
        if self.transform:
            image = self.transform(image)
        if not self.args.inference :
            label = self.all_item[item_id][1]
            label = torch.from_numpy(np.array(label))
            return {"image": image, "label": label, "names": image_name}
        else:
            return {"image": image, "names": image_name}


def make_transform(mode):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if mode == "train":
        return T.Compose([
            T.Aug(),
            normalize,
        ]
        )
    if mode == "val" or mode == "inference":
        return T.Compose([
            normalize,
        ]
        )
    raise ValueError(f'unknown {mode}')


# def get_args_parser():
#     parser = argparse.ArgumentParser('Set 3D model', add_help=False)
#     parser.add_argument('--csv_root', default='../data_profile.csv',
#                         help='path csv record')
#     parser.add_argument('--dataset_dir', default='/home/wbw/PAN/city_analysis/',
#                         help='path for save data')
#     parser.add_argument("--multi_label", default=True, type=bool)
#     return parser
#
#
# parser = argparse.ArgumentParser('3D model training and evaluation script', parents=[get_args_parser()])
# args = parser.parse_args()
# train, val = MakeList(args).read_csv()
# print(val)