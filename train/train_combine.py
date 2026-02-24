import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import json, argparse, random
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision.models as model
import arguments, utils
from models.ensemble import Ensemble
from distillation import Linf_PGD, Linf_distillation, Constraint_PGD, Onmanifold_PGD,HighFrequenceEdit,HighFrequenceMask
import copy
import numpy as np
from wideresnet import WideResNet

class DVERGE_Trainer():
    def __init__(self, base_models,train_models, optimizers, schedulers,
                 trainloader, testloader,
                 writer, save_root=None, **kwargs):
        self.models = train_models
        # 拷贝一份基准模型
        self.criterion_models = base_models
        self.epochs = 200
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.trainloader = trainloader
        self.testloader = testloader

        self.writer = writer
        self.save_root = save_root
        
        self.criterion = nn.CrossEntropyLoss()

        # distillation configs
        self.distill_fixed_layer = kwargs['distill_fixed_layer']
        self.fb =  kwargs['fb']
        self.numclass = kwargs['numclass']
        self.batchsize = kwargs['batch_size']
        self.distill_cfg = {'eps': kwargs['distill_eps'], 
                           'alpha': kwargs['distill_alpha'],
                           'steps': kwargs['distill_steps'],
                        #    'layer': kwargs['distill_layer'],
                        #    'rand_start': kwargs['distill_rand_start'],
                        #    'before_relu': True,
                           'momentum': kwargs['distill_momentum'],
                            'is_targeted': True,
                          }
        self.arch = kwargs['arch'],
        self.dateset =  kwargs['dateset'] ,
        self.distill_cfg1 = {'eps': kwargs['distill_eps'], 
                           'alpha': kwargs['distill_alpha'],
                           'steps': kwargs['distill_steps'],
                           'layer': kwargs['distill_layer'],
                           'rand_start': kwargs['distill_rand_start'],
                           'before_relu': True,
                           'momentum': kwargs['distill_momentum']
                          }
        
        # diversity training configs
        self.plus_adv = kwargs['plus_adv']
        self.coeff = kwargs['dverge_coeff']
        if self.plus_adv:
            self.attack_cfg = {'eps': kwargs['eps'], 
                               'alpha': kwargs['alpha'],
                               'steps': kwargs['steps'],
                               'is_targeted': False,
                               'rand_start': True
                              }
        self.depth = kwargs['depth']
    
    def get_epoch_iterator(self):
        iterator = tqdm(list(range(1,self.epochs+1)), total=self.epochs, desc='Epoch',
                        leave=True, position=1)
        return iterator
    
    def get_batch_iterator(self):
        loader = utils.DistillationLoader(self.trainloader, self.trainloader)
        iterator = tqdm(loader, desc='Batch', leave=False, position=2)
        return iterator

    def run(self):
        epoch_iter = self.get_epoch_iterator()
        for epoch in epoch_iter:
            self.train(epoch)
            self.test(epoch)
            if epoch%5 == 0:
                self.save(epoch)

    def train(self, epoch):
        for m in self.models:
            m.train()
        # edit = HighFrequenceMask(seed=15011)
        if not self.distill_fixed_layer:
            tqdm.write('Randomly choosing a layer for distillation...')
            self.distill_cfg['layer'] = random.randint(1, self.depth)

        losses = [0 for i in range(len(self.models))]
        
        batch_num = np.int32(np.floor(50000/self.batchsize))
        k = self.numclass
        random_idx = np.random.permutation(batch_num)
        part_size = np.int32(np.floor(batch_num/k))
        batch_iter = self.get_batch_iterator()
        for batch_idx, (si, sl, ti, tl) in enumerate(batch_iter):
            si, sl = si.cuda(), sl.cuda()
            ti, tl = ti.cuda(), tl.cuda()
            # a =  self.models[0](si)==self.criterion_models[0](si)
            # print(a)
            if self.plus_adv:
                adv_inputs_list = []
            
            distilled_data_list = []
            for idx, z in enumerate(self.models):
                curr_k =torch.from_numpy(np.int32(np.floor( np.where(random_idx==batch_idx)[0]/part_size ))).cuda()
                temp = Constraint_PGD(self.fb,self.criterion_models[0], si, (sl+curr_k+idx)%self.numclass, **self.distill_cfg)

                distilled_data_list.append(temp)

                # if self.plus_adv:
                #     temp = Linf_PGD(m, si, sl, **self.attack_cfg)
                #     adv_inputs_list.append(temp)

            for i, m in enumerate(self.models):
                loss = 0

                for j, distilled_data in enumerate(distilled_data_list):
                    if i == j:
                        continue

                    outputs = m(distilled_data)
                    loss += self.criterion(outputs, sl)
                
                if self.plus_adv:
                    outputs = m(adv_inputs_list[i])
                    loss = self.coeff * loss + self.criterion(outputs, sl)

                losses[i] += loss.item()
                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()            

        for i in range(len(self.models)):
            self.schedulers[i].step()

        print_message = 'Epoch [%3d] | ' % epoch
        for i in range(len(self.models)):
            print_message += 'Model{i:d}: {loss:.4f}  '.format(
                i=i+1, loss=losses[i]/(batch_idx+1))
        tqdm.write(print_message)

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
        torch.save(state_dict, os.path.join(self.save_root,'arch{}_eps{}alpha{}_{}_fb{}_epoch{}.pth'.format(self.arch,self.distill_cfg['eps'],self.distill_cfg['alpha'],self.dateset ,self.fb,epoch ) ))


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR10 DVERGE Training of Ensemble', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.dverge_train_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # set up writer, logger, and save directory for models
    save_root = os.path.join('checkpoint_wrn-res', 
        'dverge', 'seed_{:d}'.format(args.seed), '{:d}_{:s}{:d}_eps_{:.2f}'.format(
            args.model_num, args.arch, args.depth, args.distill_eps), 'step_{}_alpha{}'.format(args.distill_steps, args.distill_alpha)
    )
    if args.distill_fixed_layer:
        save_root += '_fixed_layer_{:d}'.format(args.distill_layer)
    if args.plus_adv:
        save_root += '_plus_adv_coeff_{:.1f}'.format(args.dverge_coeff)
    if args.start_from == 'scratch':
        save_root += '_start_from_scratch'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    else:
        print('*********************************')
        print('* The checkpoint already exists *')
        print('*********************************')

    writer = SummaryWriter(save_root.replace('checkpoints', 'runs'))

    # dump configurations for potential future references
    with open(os.path.join(save_root, 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)
    with open(os.path.join(save_root.replace('checkpoints', 'runs'), 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)

    # set up random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # initialize models
    if args.start_from == 'baseline':
        args.model_file = os.path.join('checkpoints', 'baseline', 'seed_0', '{:d}_{:s}{:d}'.format(args.model_num, args.arch, args.depth), 'epoch_200.pth')
    elif args.start_from == 'scratch':
        print("scratch")
        args.model_file = None
        
    if args.arch.lower() == 'resnet':
        model_file = '/data/zwl/zwl/DVERGE_code/checkpoints/baseline/seed_0/3_ResNet20/epoch_200.pth'
    elif  args.arch.lower() == 'wrn': 
        model_file = '/remote-home/ideven/DVERGE_tiny/checkpoints_518/baseline_c10/seed_0/3_WRN20/res_vanilla_cifa100_epoch_200.pth'
        pgd_file = '/remote-home/ideven/DVERGE_tiny/checkpoints/epoch_200.pth'
    else:
        model_file = '/data/zwl/zwl/DVERGE_code/checkpoints/baseline/seed_0/3_ResNet20/epoch_200.pth'

    if args.dateset == 'cifar10': 
        args.numclass = 10
    elif  args.dateset == 'cifar100':
        args.numclass = 100
    else:
        args.numclass = 200
    args.arch = 'resnet'
    base_models = utils.get_models(args,train=False, as_ensemble=False, model_file=pgd_file)
    
    # train_models = []
    # model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,args.widen_factor, dropRate=args.droprate)
    # device = torch.device("cuda")
    # for i in range(args.model_num):
    #     train_models.append(  WideResNet( depth = 28, num_classes = 10 , widen_factor = 10 , dropRate=0.0).to(device))
    # get data loaders
    trainloader, testloader = utils.get_loaders(args)
    args.arch = 'WRN'
    train_models = utils.get_models(args, train=True, as_ensemble=False , model_file=model_file)
    # get optimizers and schedulers
    optimizers = utils.get_optimizers(args, train_models)
    schedulers = utils.get_schedulers(args, optimizers)

    # train the ensemble
    trainer = DVERGE_Trainer(base_models,train_models, optimizers, schedulers, trainloader, testloader, writer, save_root, **vars(args))
    trainer.run()


if __name__ == '__main__':
    main()
