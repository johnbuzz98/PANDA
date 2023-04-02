import logging
import wandb
import time
import os
import json
import torch
from sklearn.metrics import roc_auc_score
from collections import OrderedDict
import utils
from losses import CompactnessLoss, EWCLoss
import tqdm
from typing import Optional

_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fit(
    model, trainloader, testloader, optimizer,
    epochs: int, savedir: str, log_interval: int, device: str, ewc:bool,
) -> None:
    step = 0
    best_auroc = 0
    ewc_loss = None

    # 초기 feature space의 중심점 설정 for Compactness Loss
    model.eval()
    auc, feature_space = get_score(model, device, trainloader, testloader)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))

    center = torch.FloatTensor(feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))  

    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        running_loss = run_epoch(model, trainloader, criterion, optimizer, log_interval, device, ewc, ewc_loss)
        auc, feature_space = get_score(model, device, trainloader, testloader)

        # wandb
        metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        metrics.update([(k, v) for k, v in running_loss.items()])
        metrics.update([("AUROC", auc)])
        wandb.log(metrics, step=step)

        step += 1

        # checkpoint
        if best_auroc < auc:
            # save results
            state = {'best_epoch':epoch, 'best_auroc':auc}
            json.dump(state, open(os.path.join(savedir, f'best_results.json'),'w'), indent=4)

            # save model
            torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
            
            _logger.info('Best AUROC {0:.3%} to {1:.3%}'.format(best_auroc, auc))

            best_acc = auc

    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(state['best_auroc '], state['best_epoch']))



def run_epoch(model, dataloader, criterion, optimizer, log_interval: int, device: str, ewc:bool, ewc_loss: Optional[float]) -> dict:   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    running_loss = AverageMeter()
    
    end = time.time()
    for idx, (inputs, targets) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        _, features = model(inputs)
        loss = criterion(features) 

        if ewc:
            loss += ewc_loss(model)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3) #gradient explosion 방지

        optimizer.step()
         
        running_loss.update(loss.item())

        
        batch_time_m.update(time.time() - end)
    
        if idx % log_interval == 0 and idx != 0: 
            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        idx+1, len(dataloader), 
                        loss       = running_loss, 
                        lr         = optimizer.param_groups[0]['lr'],
                        batch_time = batch_time_m,
                        rate       = inputs.size(0) / batch_time_m.val,
                        rate_avg   = inputs.size(0) / batch_time_m.avg,
                        data_time  = data_time_m))
   
        end = time.time()
    
    return OrderedDict([('loss',running_loss.avg)])
        
def get_score(model, device, trainloader, testloader):
    train_feature_space = []
    with torch.no_grad():
        for i in trainloader:
            imgs = i[0].to(device)
            _, features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    with torch.no_grad():
        for i in testloader:
            imgs = i[0].to(device)
            _, features = model(imgs)
            test_feature_space.append(features)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = testloader.dataset.targets

    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space
    
                
