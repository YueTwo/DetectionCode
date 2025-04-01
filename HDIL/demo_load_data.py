from imageio import imread
import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

DATABASE = '/root/HDIL/imagesFromFE'


class DatasetInstance:
    def __init__(self, img_size, map_size, norm):
        self.img_size = img_size
        self.map_size = map_size

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(*norm)
        ])

        self.transform_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(map_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        files = os.listdir(DATABASE)

        self.images = ['{0}/{1}'.format(DATABASE, _) for _ in files]

        print('Constructed Dataset `{0}` of size `{1}`'.format(DATABASE, self.__len__()))

    def load_image(self, path):
        return self.transform(imread(path))

    def load_mask(self, path):
        return self.transform_mask(imread(path))

    def __getitem__(self, index=0):
        im_name = self.images[index]
        img = self.load_image(im_name)
        return {'img': img, 'im_name': im_name}

    def __len__(self):
        return len(self.images)
