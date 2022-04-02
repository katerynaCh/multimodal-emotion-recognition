'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import csv
import torch
import shutil
import numpy as np
import sklearn


class AverageMeter(object):
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


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()



def calculate_accuracy(output, target, topk=(1,), binary=False):
    """Computes the precision@k for the specified values of k"""
    
    maxk = max(topk)
    #print('target', target, 'output', output)    
    if maxk > output.size(1):
        maxk = output.size(1)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #print('Target: ', target, 'Pred: ', pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        if k > maxk:
            k = maxk
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    if binary:
        #print(list(target.cpu().numpy()),  list(pred[0].cpu().numpy()))
        f1 = sklearn.metrics.f1_score(list(target.cpu().numpy()),  list(pred[0].cpu().numpy()))
        #print('F1: ', f1)
        return res, f1*100
    #print(res)
    return res


def save_checkpoint(state, is_best, opt, fold):
    torch.save(state, '%s/%s_checkpoint'% (opt.result_path, opt.store_name)+str(fold)+'.pth')
    if is_best:
        shutil.copyfile('%s/%s_checkpoint' % (opt.result_path, opt.store_name)+str(fold)+'.pth','%s/%s_best' % (opt.result_path, opt.store_name)+str(fold)+'.pth')


def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(opt.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
        #param_group['lr'] = opt.learning_rate


