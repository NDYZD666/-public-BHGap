from models import models_vit
import sys
import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.Generate_Model import GenerateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import datetime
from dataloader.video_dataloader import train_data_loader, test_data_loader
from sklearn.metrics import confusion_matrix
import tqdm
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random

seed = 1
random.seed(seed)  
np.random.seed(seed) 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)

    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=8)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int)

    parser.add_argument('--exper-name', type=str)
    parser.add_argument('--temporal-layers', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=224)

    parser.add_argument('--adv-weight', type=float, default=0.2, 
                        help='Weight for adversarial loss')
    parser.add_argument('--mmd-weight', type=float, default=0.1,
                        help='Weight for MMD loss')

    args = parser.parse_args()
    return args

def main(set, args):
    
    data_set = set+1
    
    if args.dataset == "DFEW":
        print("*********** DFEW Dataset Fold  " + str(data_set) + " ***********")
        log_txt_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/' + 'log.txt'
        log_curve_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/' + 'log.png'
        log_confusion_matrix_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/' + 'cn.png'
        checkpoint_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/'+'checkpoint/'+'model.pth'
        best_war_checkpoint_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/'+'checkpoint/'+'model_best_war.pth'
        best_uar_checkpoint_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/'+'checkpoint/'+'model_best_uar.pth'
        train_annotation_file_path = "./annotation/DFEW_set_"+str(data_set)+"_train.txt"
        test_annotation_file_path = "./annotation/DFEW_set_"+str(data_set)+"_test.txt"
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/')
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/checkpoint/')
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/code/')
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/code/models')
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/code/AudioMAE')
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/code/dataloader')

        for filename in ['main.py', 'train_DFEW.sh', 'train_MAFW.sh', 'models/fusion_gate.py','models/Generate_Model.py', 'models/prompt_blocks.py', 'dataloader/video_dataloader.py', 'dataloader/video_transform.py', 'models/models_vit.py', 'AudioMAE/audio_models_vit.py']:
            shutil.copyfile(filename, './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/code/'+filename)

    elif args.dataset == "MAFW":
        print("*********** MAFW Dataset Fold  " + str(data_set) + " ***********")
        log_txt_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/' + 'log.txt'
        log_curve_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/' + 'log.png'
        log_confusion_matrix_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/' + 'cn.png'
        checkpoint_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/'+'checkpoint/'+'model.pth'
        best_war_checkpoint_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/'+'checkpoint/'+'model_best_war.pth'
        best_uar_checkpoint_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/'+'checkpoint/'+'model_best_uar.pth'

        train_annotation_file_path = "./annotation/MAFW_set_"+str(data_set)+"_train_faces.txt"
        test_annotation_file_path = "./annotation/MAFW_set_"+str(data_set)+"_test_faces.txt"
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/')
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/checkpoint/')
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/code/')
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/code/models')
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/code/AudioMAE')
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/code/dataloader')

        for filename in ['main.py', 'train_DFEW.sh', 'train_MAFW.sh', 'models/fusion_gate.py','models/Generate_Model.py', 'models/prompt_blocks.py', 'dataloader/video_dataloader.py', 'dataloader/video_transform.py', 'models/models_vit.py', 'AudioMAE/audio_models_vit.py']:
            shutil.copyfile(filename, './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/code/'+filename)

    best_acc = 0
    best_uar = 0.0
    best_war = 0.0
    recorder = RecorderMeter(args.epochs)
    cls_loss_recorder = RecorderMeter(args.epochs)
    print('The training name: ' + time_str)
       
    model = GenerateModel(args=args)
    with open(log_txt_path, 'a') as f:
            f.write('The Model Struct: ' + str(model) +'\n')
            
  
    # only open learnable part
    for name, param in model.named_parameters():
        param.requires_grad = True #False

    for name, param in model.named_parameters():
        if "image_encoder" in name:
            param.requires_grad = False 
        if "audio_model" in name:
            param.requires_grad = False

        if "our_classifier" in name:
            param.requires_grad = True
        if "positional_embedding" in name:
            param.requires_grad = True
        if "learnable_prompts" in name:
            param.requires_grad = True
        if "pos_embed" in name:
            param.requires_grad = True
        if "audio_proj" in name:
            param.requires_grad = True
        if "temporal" in name:
            param.requires_grad = True
        if "gate" in name:
            param.requires_grad = True
        if "context_att" in name:
            param.requires_grad = True
        if "learnable_q" in name:
            param.requires_grad = True
        if "audio_att" in name:
            param.requires_grad = True
        if "norm_xt" in name:
            param.requires_grad = True
        if "norm_xt_2" in name:
            param.requires_grad = True
        if "norm_qs" in name:
            param.requires_grad = True
        if "cross_model_aggregator" in name:
            param.requires_grad = True
        if "audio_patch_level_conv_blocks" in name:
            param.requires_grad = True
        if "image_patch_level_conv_blocks" in name:
            param.requires_grad = True
        if "fusion_model" in name:
            param.requires_grad = True
        if "domain_classifier" in name:
            param.requires_grad = True
        

    model_parameters = model.parameters()
    model = torch.nn.DataParallel(model).cuda()

    # print params   
    print('************************')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('************************')
    
    with open(log_txt_path, 'a') as f:
        for k, v in vars(args).items():
            f.write(str(k) + '=' + str(v) + '\n')
    
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()
    
    # define optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) 
    cudnn.benchmark = True

    # Data loading code
    train_data = train_data_loader(list_file=train_annotation_file_path,
                                   num_segments=16,
                                   duration=1,
                                   image_size=args.img_size,
                                   args=args)

    test_data = test_data_loader(list_file=test_annotation_file_path,
                                 num_segments=16,
                                 duration=1,
                                 image_size=args.img_size)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    for epoch in range(0, args.epochs):

        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate_0 = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            print(inf)
            f.write('Current learning rate: ' + str(current_learning_rate_0) + '\n')
            print('Current learning rate: ', current_learning_rate_0)        
            
        # train for one epoch
        train_acc, train_los, train_cls_los, train_adv_los, train_mmd_los = train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path)

        # evaluate on validation set
        val_acc, val_los, val_uar, val_war, val_cls_los, val_adv_los, val_mmd_los = validate(val_loader, model, criterion, args, log_txt_path)
        
        # Update best metrics and save models
        is_best_uar = val_uar > best_uar
        is_best_war = val_war > best_war
        
        best_acc = max(val_acc, best_acc)
        best_uar = max(val_uar, best_uar)
        best_war = max(val_war, best_war)
        
        scheduler.step()

        if is_best_uar:
            save_checkpoint({'epoch': epoch + 1,
                           'state_dict': model.state_dict(),
                           'best_acc': best_acc,
                           'best_uar': best_uar,
                           'best_war': best_war,
                           'optimizer': optimizer.state_dict(),
                           'recorder': recorder}, 
                          best_uar_checkpoint_path)
        
        if is_best_war:
            save_checkpoint({'epoch': epoch + 1,
                           'state_dict': model.state_dict(),
                           'best_acc': best_acc,
                           'best_uar': best_uar,
                           'best_war': best_war,
                           'optimizer': optimizer.state_dict(),
                           'recorder': recorder}, 
                          best_war_checkpoint_path)

        # Print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        cls_loss_recorder.update(epoch, train_cls_los, train_cls_los, val_cls_los, val_cls_los)
        recorder.plot_curve(log_curve_path)
        cls_loss_recorder.plot_curve(log_curve_path.replace('.png', '_cls_loss.png'))

        print(f'Current UAR: {val_uar:.3f}, Best UAR: {best_uar:.3f}')
        print(f'Current WAR: {val_war:.3f}, Best WAR: {best_war:.3f}')
        print('The best accuracy: {:.3f}'.format(best_acc.item()))
        print('An epoch time: {:.2f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write(f'Current UAR: {val_uar:.3f}, Best UAR: {best_uar:.3f}\n')
            f.write(f'Current WAR: {val_war:.3f}, Best WAR: {best_war:.3f}\n')
            f.write('The best accuracy: {:.3f}\n'.format(best_acc.item()))
            f.write('An epoch time: {:.2f}s\n'.format(epoch_time))

    # Final evaluation with both best models
    print("Evaluating best UAR model...")
    final_uar_from_uar_model, final_war_from_uar_model = computer_uar_war(val_loader, model, best_uar_checkpoint_path, 
                                                                          log_confusion_matrix_path.replace('.png', '_best_uar.png'), 
                                                                          log_txt_path, data_set, args.class_names, "UAR")
    
    print("Evaluating best WAR model...")
    final_uar_from_war_model, final_war_from_war_model = computer_uar_war(val_loader, model, best_war_checkpoint_path, 
                                                                          log_confusion_matrix_path.replace('.png', '_best_war.png'), 
                                                                          log_txt_path, data_set, args.class_names, "WAR")
    
    with open(log_txt_path, 'a') as f:
        f.write('========================\n')
        f.write(f'Best UAR Model - UAR: {final_uar_from_uar_model:.2f}, WAR: {final_war_from_uar_model:.2f}\n')
        f.write(f'Best WAR Model - UAR: {final_uar_from_war_model:.2f}, WAR: {final_war_from_war_model:.2f}\n')
        f.write('========================\n')
    
    return final_uar_from_uar_model, final_war_from_uar_model


def train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    cls_losses = AverageMeter('Cls Loss', ':.4f')
    adv_losses = AverageMeter('Adv Loss', ':.4f')
    mmd_losses = AverageMeter('MMD Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    
    progress = ProgressMeter(len(train_loader),
                             [losses, cls_losses, adv_losses, mmd_losses, top1],
                             prefix="Epoch: [{}]".format(epoch),
                             log_txt_path=log_txt_path)
    
    model.train()

    for i, (images, target, audio) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda()
        audio = audio.cuda()
        batch_size = images.size(0)

        cls_output, adv_loss, mmd_loss = model(images, audio)
        
        cls_loss = criterion(cls_output, target)
        
        if isinstance(adv_loss, torch.Tensor) and adv_loss.dim() > 0:
            adv_loss = adv_loss.mean()
        if isinstance(mmd_loss, torch.Tensor) and mmd_loss.dim() > 0:
            mmd_loss = mmd_loss.mean()
        
        total_loss = cls_loss + args.adv_weight * adv_loss + args.mmd_weight * mmd_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        cls_losses.update(cls_loss.item(), batch_size)
        adv_losses.update(adv_loss.item(), batch_size)
        mmd_losses.update(mmd_loss.item(), batch_size)
        losses.update(total_loss.item(), batch_size)
        
        acc1, _ = accuracy(cls_output, target, topk=(1, 5))
        top1.update(acc1[0], batch_size)

        if i % args.print_freq == 0:
            progress.display(i)

    with open(log_txt_path, 'a') as f:
        f.write(f'Epoch {epoch} Summary:\n')
        f.write(f'  Avg Cls Loss: {cls_losses.avg:.4f}\n')
        f.write(f'  Avg Adv Loss: {adv_losses.avg:.4f}\n')
        f.write(f'  Avg MMD Loss: {mmd_losses.avg:.4f}\n')
        f.write(f'  Avg Total Loss: {losses.avg:.4f}\n')
        f.write(f'  Avg Accuracy: {top1.avg:.3f}\n')
        
    return top1.avg, losses.avg, cls_losses.avg, adv_losses.avg, mmd_losses.avg



def validate(val_loader, model, criterion, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    cls_losses = AverageMeter('Cls Loss', ':.4f')
    adv_losses = AverageMeter('Adv Loss', ':.4f')
    mmd_losses = AverageMeter('MMD Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    
    progress = ProgressMeter(len(val_loader),
                             [losses, cls_losses, adv_losses, mmd_losses, top1],
                             prefix='Test: ',
                             log_txt_path=log_txt_path)

    model.eval()

    all_predicted = []
    all_targets = []

    with torch.no_grad():
        for i, (images, target, audio) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            audio = audio.cuda()
            
            output, adv_loss, mmd_loss = model(images, audio)
            cls_loss = criterion(output, target)
            
            if isinstance(adv_loss, torch.Tensor) and adv_loss.dim() > 0:
                adv_loss = adv_loss.mean()
            if isinstance(mmd_loss, torch.Tensor) and mmd_loss.dim() > 0:
                mmd_loss = mmd_loss.mean()
            
            total_loss = cls_loss + args.adv_weight * adv_loss + args.mmd_weight * mmd_loss
            
            cls_losses.update(cls_loss.item(), images.size(0))
            adv_losses.update(adv_loss.item(), images.size(0))
            mmd_losses.update(mmd_loss.item(), images.size(0))
            losses.update(total_loss.item(), images.size(0))
            
            acc1, _ = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))

            _, predicted = torch.max(output, 1)
            all_predicted.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                progress.display(i)

        cm = confusion_matrix(all_targets, all_predicted)
        war = 100.0 * np.sum(np.diag(cm)) / np.sum(cm)
        uar = 100.0 * np.mean(np.diag(cm) / (cm.sum(axis=1) + 1e-5))
        
        print('Current Accuracy: {top1.avg:.3f}, UAR: {uar:.3f}, WAR: {war:.3f}'.format(top1=top1, uar=uar, war=war))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.3f}, UAR: {uar:.3f}, WAR: {war:.3f}\n'.format(top1=top1, uar=uar, war=war))
            f.write(f'Val Cls Loss: {cls_losses.avg:.4f}, Adv Loss: {adv_losses.avg:.4f}, MMD Loss: {mmd_losses.avg:.4f}\n')
    
    return top1.avg, losses.avg, uar, war, cls_losses.avg, adv_losses.avg, mmd_losses.avg


def save_checkpoint(state, checkpoint_path):
    torch.save(state, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(self.log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)


def plot_confusion_matrix(cm, classes, normalize=True, title='confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        fmt = '.2f'
    else:
        fmt = 'd'
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt) + '%' if normalize else format(cm[i, j], fmt), 
                 fontsize=12,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

def computer_uar_war(val_loader, model, checkpoint_path, log_confusion_matrix_path, log_txt_path, data_set, class_names, model_type=""):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()

    all_predicted = []
    all_targets = []
    total_adv_loss = 0.0
    total_mmd_loss = 0.0
    samples = 0
    
    with torch.no_grad():
        for i, (images, target, audio) in enumerate(tqdm.tqdm(val_loader)):
            images = images.cuda()
            target = target.cuda()
            audio = audio.cuda()
            
            output, adv_loss, mmd_loss = model(images, audio)
            
            if isinstance(adv_loss, torch.Tensor) and adv_loss.dim() > 0:
                adv_loss = adv_loss.mean()
            if isinstance(mmd_loss, torch.Tensor) and mmd_loss.dim() > 0:
                mmd_loss = mmd_loss.mean()
            
            batch_size = images.size(0)
            total_adv_loss += adv_loss.item() * batch_size
            total_mmd_loss += mmd_loss.item() * batch_size
            samples += batch_size
            
            predicted = output.argmax(dim=1)
            all_predicted.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_adv_loss = total_adv_loss / samples
    avg_mmd_loss = total_mmd_loss / samples

    cm = confusion_matrix(all_targets, all_predicted)
    war = 100.0 * np.sum(np.diag(cm)) / np.sum(cm)
    uar = 100.0 * np.mean(np.diag(cm) / (cm.sum(axis=1) + 1e-5))
    
    plt.figure(figsize=(10, 8))
    title_suffix = f" (Best {model_type} Model)" if model_type else ""
    title_ = f"Confusion Matrix on {args.dataset} fold {data_set}{title_suffix} (%)"
    plot_confusion_matrix(cm, classes=class_names, title=title_)
    plt.savefig(log_confusion_matrix_path)
    plt.close()
    
    print(f'{model_type} Model - Final UAR: {uar:.2f}, WAR: {war:.2f}')
    print(f'{model_type} Model - Final Adv Loss: {avg_adv_loss:.4f}, MMD Loss: {avg_mmd_loss:.4f}')
    
    with open(log_txt_path, 'a') as f:
        f.write('************************\n')
        f.write(f'{model_type} Model - Final UAR: {uar:.2f}, WAR: {war:.2f}\n')
        f.write(f'{model_type} Model - Final Adv Loss: {avg_adv_loss:.4f}, MMD Loss: {avg_mmd_loss:.4f}\n')
        f.write('************************\n')
    
    return uar, war


if __name__ == '__main__':
    args = parse_args() 
    UAR = 0.0
    WAR = 0.0
    now = datetime.datetime.now()
    time_str = now.strftime("%y%m%d%H%M")
    time_str = time_str + args.exper_name

    print('************************')
    for k, v in vars(args).items():
        print(k,'=',v)
    print('************************')

    if args.dataset == "DFEW":
        args.number_class = 7
        args.class_names = [
    'happiness.',
    'sadness.',
    'neutral.',
    'anger.',
    'surprise.',
    'disgust.',
    'fear.'
        ]

        all_fold = 5
    elif args.dataset == "MAFW":
        all_fold = 5
        args.number_class = 11
        args.class_names = ["1", '2', '3', '4','5', '6', '7', '8', '9', '10', '11']

    for set in range(all_fold):
        uar, war = main(set, args)
        UAR += float(uar)
        WAR += float(war)
        
    print('********* Final Results *********')   
    print("UAR: %0.2f" % (UAR/all_fold))
    print("WAR: %0.2f" % (WAR/all_fold))
    print('*********************************')