# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from timm.data import create_transform

from continual_datasets.continual_datasets import *

import utils
#TODO check softmax applied correctly.
class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.dataset.startswith('Split-'):
        dataset_train, dataset_val = get_dataset(args.dataset.replace('Split-',''), transform_train, transform_val, args)

        args.nb_classes = len(dataset_val.classes)

        splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
    else:
        if args.dataset == '5-datasets':
            dataset_list = ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST']
        else:
            dataset_list = args.dataset.split(',')
        
        if args.shuffle:
            random.shuffle(dataset_list)
    
        args.nb_classes = 0

    for i in range(args.num_tasks):
        if args.dataset.startswith('Split-'):
            dataset_train, dataset_val = splited_dataset[i]

        else:
            dataset_train, dataset_val = get_dataset(dataset_list[i], transform_train, transform_val, args)

            transform_target = Lambda(target_transform, args.nb_classes)

            if class_mask is not None:
                class_mask.append([i + args.nb_classes for i in range(len(dataset_val.classes))])
                args.nb_classes += len(dataset_val.classes)

            if not args.task_inc:
                dataset_train.target_transform = transform_target
                dataset_val.target_transform = transform_target
        
        if args.distributed and utils.get_world_size() > 1:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})

    return dataloader, class_mask

def get_dataset(dataset, transform_train, transform_val, args,):
    if dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'FashionMNIST':
        dataset_train = FashionMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = FashionMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'NotMNIST':
        dataset_train = NotMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = NotMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'Flower102':
        dataset_train = Flowers102(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = Flowers102(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'Cars196':
        dataset_train = StanfordCars(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = StanfordCars(args.data_path, split='test', download=True, transform=transform_val)
        
    elif dataset == 'CUB200':
        dataset_train = CUB200(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = CUB200(args.data_path, train=False, download=True, transform=transform_val).data
    
    elif dataset == 'Scene67':
        dataset_train = Scene67(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Scene67(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'TinyImagenet':
        dataset_train = TinyImagenet(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = TinyImagenet(args.data_path, train=False, download=True, transform=transform_val).data
        
    elif dataset == 'Imagenet-R':
        dataset_train = Imagenet_R(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Imagenet_R(args.data_path, train=False, download=True, transform=transform_val).data
    
    elif dataset == 'ICMem':
        from ICMem import ICMemDataset
        dataset_train = ICMemDataset(args.data_path, train=True, transform=transform_train, baseline_file=args.baseline_file)
        dataset_val = ICMemDataset(args.data_path, train=False, transform=transform_val)
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val

def split_single_dataset(dataset_train, dataset_val, args):
    nb_classes = len(dataset_val.classes)
    split_datasets = list()
    mask = list()
    if args.replay_size > 0:
        from collections import defaultdict
        replay_buffer = defaultdict(list)  # Stores class-wise images
    else:
        replay_buffer = None

     # Check if we're working with ICMem dataset (requires 2-1-1-1 split)
    if args.dataset == "Split-ICMem":
        assert nb_classes == 5, "Expected 5 total classes for 2-1-1-1 split."
        assert args.num_tasks == 4, "Expected 4 tasks for 2-1-1-1 split."

        # Define custom class distribution per task
        class_splits = [2, 1, 1, 1]

    else:
        assert nb_classes % args.num_tasks == 0, "Number of classes must be evenly divisible by num_tasks."
        classes_per_task = nb_classes // args.num_tasks
        class_splits = [classes_per_task] * args.num_tasks  # Equal split
    
    labels = list(range(nb_classes))  # Default class order

    if args.shuffle:
        random.shuffle(labels)

    total_classes = 0
    for num_classes in class_splits:
        train_split_indices = []
        test_split_indices = []
        
        scope = labels[:num_classes]
        labels = labels[num_classes:] # Remove assigned classes from the list

        if args.dataset == "Split-ICMem" and total_classes > 0:
            mask.append(mask[-1] + scope)
        else:
            mask.append(scope)

        # Important splitting logic
        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)
                
        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)
        
        if args.replay_size > 0:
            replay_size = args.replay_size
            total_classes += num_classes
            base_per_class = replay_size // total_classes  # Integer division
            extra_per_class = replay_size % total_classes  # Compute remainder

            if replay_buffer:
                replay_indices = []
                # Collect all replay images from the buffer
                for past_class, indices in replay_buffer.items():
                    replay_indices.extend(indices)  # Just add whatever is stored
                # Add replay images to training dataset
                train_split_indices.extend(replay_indices)
            
            for class_id in scope:
                class_indices = [i for i in train_split_indices if dataset_train.targets[i] == class_id]
                replay_buffer[class_id].extend(random.sample(class_indices, base_per_class))

            if sum(len(v) for v in replay_buffer.values()) > replay_size:
                for i, past_class in enumerate(replay_buffer.keys()):
                    sample_size = base_per_class + (1 if i < extra_per_class else 0)  # Distribute remainder
                    replay_buffer[past_class] = random.sample(replay_buffer[past_class], sample_size)
        random.shuffle(train_split_indices)
        random.shuffle(test_split_indices)
        
        subset_train, subset_val =  Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)
        split_datasets.append([subset_train, subset_val])
        print(f"Task {len(split_datasets)}")
        print(f"Training Samples: {len(train_split_indices)}")
        print(f"Validation Samples: {len(test_split_indices)}")
        print(f"Replay Buffer Class Distribution: { {k: len(v) for k, v in replay_buffer.items()} if replay_buffer else 'N/A'}\n")

    
    return split_datasets, mask

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    
    return transforms.Compose(t)
