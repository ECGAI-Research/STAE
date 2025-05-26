import os
import random
import argparse
import time
import torch
import numpy as np
from tqdm import tqdm
from utils import normalize
from dataloader import TestSet
from sklearn.metrics import roc_auc_score, roc_curve  , precision_recall_curve, accuracy_score , recall_score,f1_score , precision_score
import copy
from lib.STAE import STAE
import seaborn as sns
import warnings 
import time

def main(args):
   
    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    else:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
    print("args: ", args)
    # Prepare DataLoader for test dataset

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    dtset = TestSet(folder=args.data_path)
    test_loader = torch.utils.data.DataLoader(dtset, batch_size=1, shuffle=False, **kwargs)
    labels = np.load(os.path.join(args.data_path, 'label.npy'))
    # Load pre-trained model 

    model = STAE(enc_in=args.dims).to(device)
    if args.load_model == 1:
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    # Perform detection and compute evaluation metrics
    accu,score= detection_test(args, model, test_loader, labels)
    test_labels = np.array(labels).astype(int)
    thershold=Find_youden_threshold(test_labels, score)
    
    compute_metrics(labels,score,thershold)    
    

# Find optimal threshold based on Youden's index (for ROC curve)

def Find_youden_threshold(y_true, y_scores):
    
    
    FPR, TPR, thresholds = roc_curve(y_true, y_scores)
    
    TNR = 1 - FPR
    
    youden_j = TPR + TNR - 1
    
    
    optimal_threshold = thresholds[np.argmax(youden_j)]
    
    
    print(f"Youden threshold : {optimal_threshold}")
    
    return optimal_threshold

# Eevaluation mode  
def detection_test(args, model, test_loader, labels):
    torch.zero_grad = True
    model.eval()
    result = []
    inference_times = []  
    for i, (time_ecg, spectrogram_ecg) in tqdm(enumerate(test_loader)):
        instance_result = []
        time_length = time_ecg.shape[1]
       
        for j in range(100 // args.mask_ratio_time):
            patch_interval_time = 4800 // args.mask_ratio_time
            time_ecg = time_ecg.float().to(device)
            mask_time = copy.deepcopy(time_ecg)
            mask = torch.zeros((1, time_length, 1), dtype=torch.bool).to(device)

            for k in range(args.mask_ratio_time):
                cut_idx = 48 * j + patch_interval_time * k
                mask[:, cut_idx:cut_idx + 48] = 1
            mask_time = torch.mul(mask_time, ~mask)
            
            patch_interval_spec = 66 // args.mask_ratio_spec
            spec_ecg = spectrogram_ecg.float().to(device)
            bs, freq_dim, time_dim, dim = spec_ecg.shape
            mask_spec = copy.deepcopy(spec_ecg)
            mask = torch.zeros((bs, freq_dim, time_dim, 1), dtype=torch.bool).to(device)

            for k in range(args.mask_ratio_spec):
                 cut_idx = 1 * j + patch_interval_spec * k
                 mask[:, :, cut_idx:cut_idx + 1] = 1
            mask_spec = torch.mul(mask_spec, ~mask)
            start_time = time.time()
            (gen_time, time_var) = model(mask_time, mask_spec) # Model prediction
            torch.cuda.synchronize()  
            end_time = time.time()
                
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            epsilon = 1e-10 
            time_err =(gen_time - time_ecg) ** 2
            l_time = torch.mean(torch.exp(-time_var) * time_err + epsilon) 
            loss = l_time
 
            loss = loss.detach().cpu().numpy()
            instance_result.append(loss)

        tmp_instance_result = np.asarray(instance_result)
        result.append(tmp_instance_result.mean())
        
    test_labels = np.array(labels).astype(int)
    scores = np.asarray(result)
    
    test_labels = np.array(labels).astype(int)
    
    max_anomaly_score = scores.max()
    
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
    
    auc_result = roc_auc_score(test_labels, scores) # Compute AUC score
    print(("Detection AUC: ", round(auc_result, 3)))
    avg_inference_time = np.mean(inference_times) # Average inference time per sample
    print(f"Average Inference Time per sample: {avg_inference_time:.6f} seconds")
    return auc_result , scores

# Compute classification metrics:precision, recall, F1 score

def compute_metrics(test_labels, scores,thershold):

      predicted_labels = (scores >=thershold).astype(int)
      f1 = f1_score(test_labels, predicted_labels)
      precision = precision_score(test_labels, predicted_labels)
      recall = recall_score(test_labels, predicted_labels)
     
      print(f"Precision: {round(precision, 3)}")
      print(f"Recall: {round(recall, 3)}")
      print(f"F1 Score: {round(f1, 3)}")
            
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Testing Sparse Temporal AutoEncoder for ECG Anomaly Detection')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--dims', type=int, default=12, help='dimension of the input data')
    parser.add_argument('--load_model', type=int, default=1, help='0 for retrain, 1 for load model')
    parser.add_argument('--load_path', type=str, default='ckpt/STAE.pt')
    parser.add_argument('--mask_ratio_time', type=int, default=30, help='mask ratio for self-restoration in time branch')
    parser.add_argument('--mask_ratio_spec', type=int, default=20, help='mask ratio for self-restoration in spectrogram branch')
    parser.add_argument('--seed', type=int, default=668, help='manual seed')
    parser.add_argument("--gpu", type=str, default="1")
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    device = "cuda:" + args.gpu if use_cuda else 'cpu'
    main(args)


