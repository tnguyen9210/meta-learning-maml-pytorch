
import pickle

from PIL import Image
import numpy as np 

import torch
import torch.nn.functional as F
import torch.utils.data as data

import torchvision.transforms as transforms


def load_data(args):

    normalize = transforms.Normalize(
            mean=[0.4914, 0.4821, 0.4463], std=[0.2467, 0.2431, 0.2611])
    
    train_transform = transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((args['img_size'], args['img_size'])),
        # transforms.RandomCrop(32, 4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((args['img_size'], args['img_size'])),
        transforms.ToTensor(),
        normalize
    ])
        
    # load data
    train_lbl = ImageDataset(ds_name='train_lbl', transform=train_transform, args=args)
    dev_lbl = ImageDataset(ds_name='dev_lbl', transform=test_transform, args=args)
    test_lbl = ImageDataset(ds_name='test_lbl', transform=test_transform, args=args)

    print(f"num_data")
    print(f"train lbl: {train_lbl.num_data}")
    print(f"dev lbl: {dev_lbl.num_data}")
    print(f"test lbl: {test_lbl.num_data}")

    train_lbl_sampler = data.RandomSampler(
        train_lbl, replacement=True, num_samples=args['num_episodes']*args['num_iters'])
    
    train_lbl = data.DataLoader(
        train_lbl, batch_size=args['num_episodes'], num_workers=16,
        sampler=train_lbl_sampler)
    dev_lbl = data.DataLoader(
        dev_lbl, batch_size=1, shuffle=False, num_workers=16)
    test_lbl = data.DataLoader(
        test_lbl, batch_size=1, shuffle=False, num_workers=16)

    print(f"num_episodes")
    print(f"train lbl: {len(train_lbl)}")
    print(f"dev lbl: {len(dev_lbl)}")
    print(f"test lbl: {len(test_lbl)}")

    return train_lbl, dev_lbl, test_lbl


class ImageDataset(data.Dataset):
    def __init__(self, ds_name, transform, args):

        # params
        self.data_dir = args['data_dir']
        self.img_size = args['img_size']
        self.nway = args['nway']
        self.sshot = args['sshot']
        self.qshot = args['qshot']
        self.support_size = self.nway*self.sshot
        self.query_size = self.nway*self.qshot
        self.num_episodes = args['num_episodes']
        self.transform = transform

        self.label2id = {}
        self.imgs = []
        c = 0
        with open(f"{self.data_dir}/{ds_name}_labels.csv") as fin:
            lines = fin.read().split('\n')[1:-1]
            for line in lines:
                img_file, label = line.split(',')
                if label not in self.label2id:
                    self.label2id[label] = c
                    self.imgs.append([img_file])
                    c += 1
                else:
                    self.imgs[self.label2id[label]].append(img_file)
                
        # for label in self.label2id.keys():
        #     print(f'\n--> {label}')
        #     print(self.imgs[self.label2id[label]][:2])

        self.num_classes = len(self.label2id)
        self.num_data = len(lines)
        self.create_episodes()
    
    def create_episodes(self):
        self.support_sets = []  # support set batches
        self.query_sets = []    # query set batches

        for eps in range(self.num_episodes):
            # 1. select n classes randomly
            batch_classes = np.random.choice(
                  self.num_classes, self.nway, False)  # no duplicate
            np.random.shuffle(batch_classes)

            support_set = []
            query_set = []
            
            for c in batch_classes:
                # 2. select kshot + kquery for each class
                batch_idxes = np.random.choice(
                    len(self.imgs[c]), self.sshot + self.qshot, False)
                np.random.shuffle(batch_idxes)
                # print('\n--> batch_idxes')
                # print(batch_idxes)
                
                support_idxes = np.array(batch_idxes[:self.sshot])
                support_set.append(
                    np.array(self.imgs[c])[support_idxes].tolist())
                # print('\n--> support_set')
                # print(support_idxes)
                # print(np.array(self.imgs[c])[support_idxes])
                
                query_idxes = np.array(batch_idxes[self.sshot:])
                query_set.append(
                    np.array(self.imgs[c])[query_idxes].tolist())
                # print('\n--> query_set')
                # print(query_idxes)
                # print(np.array(self.imgs[c])[query_idxes])

            # shuffle support and query sets
            # so that orders for classes are not same in the two sets
            np.random.shuffle(support_set)
            np.random.shuffle(query_set)
                
            self.support_sets.append(support_set)
            self.query_sets.append(query_set)
            
    def __getitem__(self, idx):

        support_x = torch.zeros(
            (self.support_size, 3, self.img_size, self.img_size), dtype=torch.float32)
        support_y = torch.zeros((self.support_size), dtype=torch.int64)
        cnt = 0
        for img_files in self.support_sets[idx]:
            for img_file in img_files:
                support_x[cnt] = self.transform(f"{self.data_dir}/images/{img_file}")
                support_y[cnt] = self.label2id[img_file[:9]]
                cnt += 1

        query_x = torch.zeros(
            (self.query_size, 3, self.img_size, self.img_size), dtype=torch.float32)
        query_y = torch.zeros((self.query_size), dtype=torch.int64)
        cnt = 0
        for img_files in self.query_sets[idx]:
            for img_file in img_files:
                query_x[cnt] = self.transform(f"{self.data_dir}/images/{img_file}")
                query_y[cnt] = self.label2id[img_file[:9]]
                cnt += 1

        # adjust true labels to relative ones ranging from 0 to nway
        unique_ids = np.unique(support_y)
        np.random.shuffle(unique_ids)

        support_y_rel = torch.zeros((self.support_size), dtype=torch.int64)
        query_y_rel = torch.zeros((self.query_size), dtype=torch.int64)
        for y_rel, y_true in enumerate(unique_ids):
            support_y_rel[support_y == y_true] = y_rel
            query_y_rel[query_y == y_true] = y_rel

        return support_x, support_y_rel, query_x, query_y_rel
        
    def __len__(self):
        return self.num_iters

    
