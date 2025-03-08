import os
import random
import argparse
import time
import torch
import math
import numpy as np
from tqdm import tqdm
from utils import time_string, convert_secs2time, AverageMeter
from SparseAutoLib.STAE import STAE
from dataloader import TrainSet, TestSet
from sklearn.metrics import roc_auc_score
import copy
import warnings

def main(args):
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
       
    print("args: ", args)
    
    # Data loading
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(TrainSet(folder=args.data_path), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(TestSet(folder=args.data_path), batch_size=1, shuffle=False, **kwargs)
    labels = np.load(os.path.join(args.data_path, 'label.npy'))

    model = STAE(enc_in=args.dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.pth_path is not None:
        checkpoint = torch.load(args.pth_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])   

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Training loop
    old_auc_result = 0
    epoch_time = AverageMeter()
    start_time = time.time()
    for epoch in range(args.epochs + 1):
        adjust_learning_rate(optimizer, args.lr, epoch, args)
        print(f' {epoch}/{args.epochs} ----- [{time_string()}] [Need: {convert_secs2time(epoch_time.avg * (args.epochs - epoch))}]')
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        train(args, model, epoch, train_loader, optimizer)
        auc_result = test(args, model, test_loader, labels)

        # Save the model if AUC improves
        if auc_result > old_auc_result:
            old_auc_result = auc_result
            if args.save_model:
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save_path, 'STAE.pt'))

    print("Final best AUC:", old_auc_result)

def train(args, model, epoch, train_loader, optimizer):
    model.train()
    total_losses = AverageMeter()
    for i, (time_ecg, spectrogram_ecg) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        
        # Apply time-domain masking
        time_ecg = time_ecg.float().to(device)
        mask_time = copy.deepcopy(time_ecg)
        mask = torch.zeros_like(time_ecg, dtype=torch.bool).to(device)
        patch_length = time_ecg.shape[1] // 100  
        for j in random.sample(range(100), args.mask_ratio_time):
            mask[:, j*patch_length:(j+1)*patch_length] = 1 
        mask_time = torch.mul(mask_time, ~mask)

        # Apply spectrogram masking
        spec_ecg = spectrogram_ecg.float().to(device)
        mask_spec = copy.deepcopy(spec_ecg)
        mask = torch.zeros_like(spec_ecg, dtype=torch.bool).to(device)
        for j in random.sample(range(66), args.mask_ratio_spec):
            mask[:, :, j:j+1, :] = 1 
        mask_spec = torch.mul(mask_spec, ~mask)

        # Forward pass
        gen_time, time_var = model(mask_time, mask_spec)

        # Loss computation
        time_err = (gen_time - time_ecg) ** 2
        loss = torch.mean(torch.exp(-time_var) * time_err) + torch.mean(time_var)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        total_losses.update(loss.item(), time_ecg.size(0))
       
    print(f'Train Epoch: {epoch} Total Loss: {total_losses.avg:.6f}')
 
def test(args, model, test_loader, labels):
    torch.zero_grad = True
    model.eval()
    result = []

    
    for i, (time_ecg, spectrogram_ecg) in tqdm(enumerate(test_loader)):
        instance_result = []
        time_length = time_ecg.shape[1]
        time_ecg = time_ecg.float().to(device) 

        # Apply masking in multiple patches
        for j in range(100 // args.mask_ratio_time):
            mask_time = copy.deepcopy(time_ecg).float().to(device)
            mask = torch.zeros_like(time_ecg, dtype=torch.bool).to(device)
            for k in range(args.mask_ratio_time):
                cut_idx = 48*j + (4800 // args.mask_ratio_time)*k
                mask[:, cut_idx:cut_idx+48] = 1
            mask_time = torch.mul(mask_time, ~mask)

            # Spectrogram masking
            spec_ecg = spectrogram_ecg.float().to(device)
            mask_spec = copy.deepcopy(spec_ecg)
            mask = torch.zeros_like(spec_ecg, dtype=torch.bool).to(device)
            for k in range(args.mask_ratio_spec):
                cut_idx = 1*j + (66 // args.mask_ratio_spec)*k
                mask[:, :, cut_idx:cut_idx+1, :] = 1
            mask_spec = torch.mul(mask_spec, ~mask)

            # Forward pass
            gen_time, time_var = model(mask_time, mask_spec)

            # Compute loss
            time_err = (gen_time - time_ecg) ** 2
            loss = torch.mean(torch.exp(-time_var) * time_err).detach().cpu().numpy()
            instance_result.append(loss)

        # Compute anomaly scores
        result.append(np.mean(instance_result))

    scores = np.asarray(result)
    scores = (scores - scores.min()) / (scores.max() - scores.min())  # Normalize scores
    auc_result = roc_auc_score(labels.astype(int), scores)
    
    print(f"AUC: {round(auc_result, 3)}")
    return auc_result
 
def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Cosine learning rate decay"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
 
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Sparse Temporal AutoEncoder for ECG Anomaly Detection')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--dims', type=int, default=12)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='ckpt/')
    parser.add_argument('--mask_ratio_time', type=int, default=30)
    parser.add_argument('--mask_ratio_spec', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=668)
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("--pth_path", type=str, default=None)
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    device = f"cuda:{args.gpu}" if use_cuda else 'cpu'
    main(args)
