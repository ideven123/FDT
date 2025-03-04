import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import json, argparse
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import arguments, utils
from models.ensemble import Ensemble


class Baseline_Trainer():
    def __init__(self, models, optimizers, schedulers,
                 trainloader, testloader,
                 writer, save_root=None, **kwargs):
        self.models = models
        self.epochs = kwargs['epochs']
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.trainloader = trainloader
        self.testloader = testloader

        self.writer = writer
        self.save_root = save_root
        
        self.criterion = nn.CrossEntropyLoss()
    
    def get_epoch_iterator(self):
        iterator = tqdm(list(range(1,self.epochs+1)), total=self.epochs, desc='Epoch',
                        leave=False, position=1)
        return iterator

    def get_batch_iterator(self):
        iterator = tqdm(self.trainloader, desc='Batch', leave=False, position=2)
        return iterator

    def run(self):
        epoch_iter = self.get_epoch_iterator()
        for epoch in epoch_iter:
            self.train(epoch)
            self.test(epoch)
            if epoch%20 == 0:
                self.save(epoch)

    def train(self, epoch):
        for m in self.models:
            m.train()

        losses = [0 for i in range(len(self.models))]
        
        batch_iter = self.get_batch_iterator()
        for batch_idx, (inputs, targets) in enumerate(batch_iter):
            inputs, targets = inputs.cuda(), targets.cuda()

            for i, m in enumerate(self.models):
                loss = 0

                outputs = m(inputs)
                loss = self.criterion(outputs, targets)
                losses[i] += loss.item()

                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()            

        print_message = 'Epoch [%3d] | ' % epoch
        for i in range(len(self.models)):
            print_message += 'Model{i:d}: {loss:.4f}  '.format(
                i=i+1, loss=losses[i]/(batch_idx+1))
        tqdm.write(print_message)

        for i in range(len(self.models)):
            self.schedulers[i].step()

        loss_dict = {}
        for i in range(len(self.models)):
            loss_dict[str(i)] = losses[i]/len(self.trainloader)
        self.writer.add_scalars('train/loss', loss_dict, epoch)

    def test(self, epoch):
        for m in self.models:
            m.eval()

        ensemble = Ensemble(self.models)

        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.cuda(), targets.cuda()

                outputs = ensemble(inputs)
                loss += self.criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

                total += inputs.size(0)

        self.writer.add_scalar('test/ensemble_loss', loss/len(self.testloader), epoch)
        self.writer.add_scalar('test/ensemble_acc', 100*correct/total, epoch)

        print_message = 'Evaluation  | Ensemble Loss {loss:.4f} Acc {acc:.2%}'.format(
            loss=loss/len(self.testloader), acc=correct/total)
        tqdm.write(print_message)

    def save(self, epoch):
        state_dict = {}
        for i, m in enumerate(self.models):
            state_dict['model_%d'%i] = m.state_dict()
        if epoch%10 == 0:
            torch.save(state_dict, os.path.join(self.save_root, 'res_vanilla_cifa100_epoch_%d.pth'%epoch))


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR10 Baseline Training of Ensemble', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # set up writer, logger, and save directory for models
    save_root = os.path.join('checkpoints_518', 'baseline_c10', 'seed_{:d}'.format(args.seed), '{:d}_{:s}{:d}'.format(args.model_num, args.arch, args.depth))
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    else:
        print('*********************************')
        print('* The checkpoint already exists *')
        print('*********************************')

    writer = SummaryWriter(save_root.replace('checkpoints', 'runs'))

    # dump configurations for potential future references
    with open(os.path.join(save_root, 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)
    with open(os.path.join(save_root.replace('checkpoints', 'runs'), 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    # set up random seed
    torch.manual_seed(args.seed)

    # initialize models
    train_models = utils.get_models(args, train=True, as_ensemble=False, model_file=None)

    # load wrn
    # from wideresnet import WideResNet
    # train_models = []
    # model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,args.widen_factor, dropRate=args.droprate)
    # device = torch.device("cuda")
    # for i in range(args.model_num):
    #     train_models.append(  WideResNet( depth = 28, num_classes = 10 , widen_factor = 10 , dropRate=0.0).to(device))

    # get data loaders
    trainloader, testloader = utils.get_loaders(args)

    # get optimizers and schedulers
    optimizers = utils.get_optimizers(args, train_models)
    schedulers = utils.get_schedulers(args, optimizers)

    # train the ensemble
    trainer = Baseline_Trainer(train_models, optimizers, schedulers,
                            trainloader, testloader, writer, save_root, **vars(args))
    trainer.run()


if __name__ == '__main__':
    main()
