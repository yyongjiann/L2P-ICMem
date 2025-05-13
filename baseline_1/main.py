# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader
from engine import *
import models
import utils

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    data_loader, class_mask = build_continual_dataloader(args)

    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
    )

       # Load custom weights here
    if args.custom_weight_path:  # Check if a custom weight path is provided
        print(f"Loading custom weights from: {args.custom_weight_path}")
        checkpoint = torch.load(args.custom_weight_path, map_location=device)
        
        if 'teacher' in checkpoint:
            state_dict = checkpoint['teacher']
        else:
            state_dict = checkpoint
        
        # Remove prefixes like "backbone."
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # Remove unexpected keys
        state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

        original_model.load_state_dict(state_dict, strict=False)  # Load the state_dict from the checkpoint

        missing_keys = ["prompt.prompt", "prompt.prompt_key", "head.weight", "head.bias"]
        for key in missing_keys:
            if key in model.state_dict():
                state_dict[key] = model.state_dict()[key]  # Use the default initialization

        pos_embed_checkpoint = state_dict["pos_embed"]
        random_extra_tokens = torch.randn(1, args.top_k * args.length, pos_embed_checkpoint.shape[2], device=pos_embed_checkpoint.device) * 0.02
        pos_embed_checkpoint = torch.cat([pos_embed_checkpoint, random_extra_tokens], dim=1)
        state_dict["pos_embed"] = nn.Parameter(pos_embed_checkpoint)
        model.load_state_dict(state_dict, strict=False)  # Load the state_dict from the checkpoint

    
    original_model.to(device)
    model.to(device)  

    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False
        
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, original_model, data_loader, device, 
                                            task_id, class_mask, acc_matrix, args,)
        
        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    #args.lr = args.lr * global_batch_size / 256.0

    optimizer = create_optimizer(args, model_without_ddp)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    embeddings = model.module.prompt.prompt_key.cpu().detach().numpy()
    total_prompts = embeddings.shape[0]
    cosine_sim_matrix = cosine_similarity(embeddings)
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=range(total_prompts), columns=range(total_prompts))

    plt.figure(figsize=(10, 10))
    sns.heatmap(cosine_sim_df, cmap="coolwarm", annot=True)
    plt.title("Pairwise Cosine Similarity Matrix of Prompts")
    plt.xlabel("Prompt Index")
    plt.ylabel("Prompt Index")
    plt.savefig('icmem_output/prompt_key_matrix_0.png')
    plt.close()

    """
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=100, n_iter=1000, random_state=42)
    tsne_emb = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=range(len(tsne_emb)), cmap='tab20')
    for i, txt in enumerate(range(len(tsne_emb))):
        plt.annotate(txt, (tsne_emb[i, 0], tsne_emb[i, 1]))
    plt.savefig(f'prompt_key_0.png')
    plt.close()
    """

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp, original_model,
                    criterion, data_loader, optimizer, lr_scheduler,
                    device, class_mask, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('L2P training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_l2p':
        from configs.cifar100_l2p import get_args_parser
        config_parser = subparser.add_parser('cifar100_l2p', help='Split-CIFAR100 L2P configs')
    elif config == 'five_datasets_l2p':
        from configs.five_datasets_l2p import get_args_parser
        config_parser = subparser.add_parser('five_datasets_l2p', help='5-Datasets L2P configs')
    elif config == 'cifar10_l2p':
        from configs.cifar10_l2p import get_args_parser
        config_parser = subparser.add_parser('cifar10_l2p', help='Split-CIFAR10 L2P configs')
    elif config == 'icmem_l2p':
        from configs.icmem_l2p import get_args_parser
        config_parser = subparser.add_parser('icmem_l2p', help='Split-ICMem L2P configs')
    else:
        raise NotImplementedError
    
    get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)

