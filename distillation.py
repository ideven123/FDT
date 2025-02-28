import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from utils import Ensemble
import torch.autograd as autograd

# 计算每一个模型的对应输出关于损失的梯度[f'(x)]_y
def gradient_wrt_input(model, inputs, targets, criterion=nn.CrossEntropyLoss()):
    inputs.requires_grad = True
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    model.zero_grad()
    loss.backward()

    data_grad = inputs.grad.data
    return data_grad.clone().detach()


def gradient_wrt_feature(model, source_data, target_data, layer, before_relu, criterion=nn.MSELoss()):
    source_data.requires_grad = True
    
    out = model.get_features(x=source_data, layer=layer, before_relu=before_relu)
    target = model.get_features(x=target_data, layer=layer, before_relu=before_relu).data.clone().detach()
    
    loss = criterion(out, target)
    model.zero_grad()
    loss.backward()

    data_grad = source_data.grad.data
    return data_grad.clone().detach()


def Linf_PGD_distill(model, dat, lbl, ti, tl, eps, alpha, steps, is_targeted=False, rand_start=True, momentum=False, mu=1, criterion=nn.CrossEntropyLoss()):
    x_nat = dat.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.) # respect image bounds
    g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        # grad = gradient_wrt_input(model, x_adv, lbl, criterion)
        grad = gradient_wrt_feature(model, x_adv, ti, layer = 20, before_relu = True)
        
        with torch.no_grad():
            if momentum:
                # Compute sample wise L1 norm of gradient
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0],1,1,1)
                # Accumulate the gradient
                new_grad = mu * g + grad # calc new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            # Get the sign of the gradient
            # sign_data_grad = new_grad.sign()
            if is_targeted:
                x_adv = x_adv - alpha * new_grad # perturb the data to MINIMIZE loss on tgt class
            else:
                x_adv = x_adv + alpha * new_grad # perturb the data to MAXIMIZE loss on gt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            #x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)
    return x_adv.clone().detach()


def Linf_PGD(model, dat, lbl, eps, alpha, steps, is_targeted=False, rand_start=True, momentum=False, mu=1, criterion=nn.CrossEntropyLoss()):
    x_nat = dat.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.) # respect image bounds
    g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_input(model, x_adv, lbl, criterion)
        with torch.no_grad():
            if momentum:
                # Compute sample wise L1 norm of gradient
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0],1,1,1)
                # Accumulate the gradient
                new_grad = mu * g + grad # calc new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            # Get the sign of the gradient
            sign_data_grad = new_grad.sign()
            if is_targeted:
                x_adv = x_adv - alpha * sign_data_grad # perturb the data to MINIMIZE loss on tgt class
            else:
                x_adv = x_adv + alpha * sign_data_grad # perturb the data to MAXIMIZE loss on gt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            #x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)
    return x_adv.clone().detach()


def Linf_distillation(model, dat, target, eps, alpha, steps, layer, before_relu=True, mu=1, momentum=True, rand_start=False):
    x_nat = dat.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.) # respect image bounds
    g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_feature(model, x_adv, target, layer, before_relu)
        # fft_img = torch.fft.fft2(x_adv)
        # grad2 = autograd.grad(outputs=x_adv , inputs=fft_img)
        # grad  = torch.matmul(grad1,grad2) 
        with torch.no_grad():
            if momentum:
                # Compute sample wise L1 norm of gradient
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0],1,1,1)
                # Accumulate the gradient
                new_grad = mu * g + grad # calc new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            x_adv = x_adv - alpha * new_grad.sign() # perturb the data to MINIMIZE loss on tgt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            #x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)
    return x_adv.clone().detach()

def test_pgd(model, dat, lbl, eps, alpha, steps, is_targeted=False, rand_start=False, momentum=False, mu=1, criterion=nn.CrossEntropyLoss()):
    edit = HighFrequenceEdit(seed=15011)
    mask =  HighFrequenceMask(seed=15011)
    ## 先编辑fft
    
    # dat = pertur(dat)
    
    x_nat = dat.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = dat.clone().detach()
    x_adv = edit(x_adv)
    x_adv = torch.clamp(x_adv, 0., 1.) # respect image bounds
    # g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    # for i in range(steps):
    #     # Calculate gradient w.r.t. data
    #     grad = gradient_wrt_input(model, x_adv, lbl, criterion)
    #     # grad = gradient_wrt_feature(model, x_adv, ti, layer = 20, before_relu = True)
    #     # grad = mask(grad)
    #     with torch.no_grad():
    #         if momentum:
    #             # Compute sample wise L1 norm of gradient
    #             flat_grad = grad.view(grad.shape[0], -1)
    #             l1_grad = torch.norm(flat_grad, 1, dim=1)
    #             grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0],1,1,1)
    #             # Accumulate the gradient
    #             data_grad = mu * g + grad # calc new grad with momentum term
    #             g = data_grad
    #         else:
    #             data_grad = grad
    #         # Get the sign of the gradient
    #         sign_data_grad = data_grad.sign()
    #         if is_targeted:
    #             x_adv = x_adv - alpha * sign_data_grad # perturb the data to MINIMIZE loss on tgt class
    #         else:
    #             x_adv = x_adv + alpha * sign_data_grad # perturb the data to MAXIMIZE loss on gt class
    #         # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
    #         #x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
    #         x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
    #         # Make sure we are still in bounds
    #         x_adv = torch.clamp(x_adv, 0., 1.)
    # p = x_adv - x_nat
    # p = mask(p)
    # x_adv = x_nat + p
    return x_adv.clone().detach()


def Constraint_PGD(fb ,model, dat, lbl, eps, alpha, steps, is_targeted=False, rand_start=True, momentum=False, mu=1, criterion=nn.CrossEntropyLoss()):
    edit = HighFrequenceEdit(fb=fb)
    # mask = HighFrequenceMask(seed=15011)
    
    x_nat = dat.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.) # respect image bounds
    g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_input(model, x_adv, lbl, criterion)
        # grad = gradient_wrt_feature(model, x_adv, ti, layer = 20, before_relu = True)
        # grad = edit(x_nat,grad)
        with torch.no_grad():
            if momentum:
                # Compute sample wise L1 norm of gradient
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0],1,1,1)
                # Accumulate the gradient
                new_grad = mu * g + grad # calc new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            # Get the sign of the gradient
            sign_data_grad = new_grad.sign()
            if is_targeted:
                x_adv = x_adv - alpha * sign_data_grad # perturb the data to MINIMIZE loss on tgt class
            else:
                x_adv = x_adv + alpha * sign_data_grad # perturb the data to MAXIMIZE loss on gt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            #x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)
    
    x_adv = edit(x_nat,x_adv)
    # x_adv = torch.clamp(x_adv, 0., 1.)
    return x_adv.clone().detach()

# 用线性空间投影回原流形上？
class PCAAttack(object):
    def __init__(self, epsilon = 2, Thres = 0.5):
        self.epsilon = epsilon
        self.Thres = Thres
    def perturb(self, x, y, epsilon = 2):

        if epsilon is not None:
            epsilon = self.epsilon

        size = x.size()
        batch_size = size[0]
        vecx = x.view(batch_size,-1)
        No_feature = vecx.size()[1]
        vecxmean = torch.mean(vecx,dim = 0)
        K = vecx - vecxmean
        K = K.numpy()
        U,S,Vt = np.linalg.svd(K)
        # for i in range(No_feature):
        #     if S[i]<=self.Thres:
        #         break

        direc = Vt[-200:-1,:]
        direc = torch.from_numpy(direc)
        direc = torch.mean(direc, dim = 0)
        norm_direc = torch.norm(direc)

        x_adv = vecx + self.epsilon * torch.div(direc, norm_direc)
        if size[1] == 3:
            x_adv = torch.clamp(x_adv,-1,1)
        else:
            x_adv = torch.clamp(x_adv,0,1)
        x_adv = x_adv.view(size)
        x_adv = np.copy(x_adv)

        return x_adv
    
def Project(dat, pertube):

    size = dat.size()
    batch_size = size[0]
    vecx = dat.view(batch_size,-1)
    pertube = pertube.view(batch_size,-1)

    # vecxmean = torch.mean(vecx,dim = 0)
    # K = vecx - vecxmean
    K = vecx.clone()
    U,S,V = torch.svd(K)

    # V[:, -20:] = 0

    project_p = torch.mul(pertube, V.T)

    return project_p.view(size)

def Onmanifold_PGD(model, dat, lbl, eps, alpha, steps, is_targeted=False, rand_start=True, momentum=False, mu=1, criterion=nn.CrossEntropyLoss()):
    # release adversarial robust overfit
    edit = HighFrequenceEdit(seed=15011)
    x_nat = dat.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.) # respect image bounds
    g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_input(model, x_adv, lbl, criterion)
        # grad = gradient_wrt_feature(model, x_adv, ti, layer = 20, before_relu = True)
        # grad = edit(grad)
        with torch.no_grad():
            if momentum:
                # Compute sample wise L1 norm of gradient
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0],1,1,1)
                # Accumulate the gradient
                new_grad = mu * g + grad # calc new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            # Get the sign of the gradient
            sign_data_grad = new_grad.sign()
            if is_targeted:
                x_adv = x_adv - alpha * sign_data_grad # perturb the data to MINIMIZE loss on tgt class
            else:
                x_adv = x_adv + alpha * sign_data_grad # perturb the data to MAXIMIZE loss on gt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            #x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)
    p = x_adv - x_nat
    p = Project(dat, p)
    x_adv = x_nat + p
    return x_adv.clone().detach()

def transferable_attack(models, dat, lbl, eps, alpha, steps, is_targeted=False, rand_start=True, momentum=False, mu=1, criterion=nn.CrossEntropyLoss()):
    x_nat = dat.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.) # respect image bounds
    g = torch.zeros_like(x_adv)
    x_t = copy.deepcopy(x_adv)

    model = Ensemble(models)
    r = eps / 15
    beta = 0.05 # 需要比较小？
    m_outer = 0
    m_hat = 0

    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        x_adv = copy.deepcopy(x_t)
        g = gradient_wrt_input(model, x_adv, lbl, criterion)
        x_adv = x_adv + r * g.sign()
        x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps).clone().detach()
        tmp = x_adv.data

        for m in models:
            g = gradient_wrt_input(m, x_adv, lbl, criterion)
            flat_grad = g.view(g.shape[0], -1)
            l2_grad = torch.norm(flat_grad, 2, dim=1)
            g = g / torch.clamp(l2_grad, min=1e-12).view(g.shape[0],1,1,1)
            m_hat = mu * m_hat + g
            x_adv = (x_adv - beta * m_hat).clone().detach()

        g = x_adv - tmp
        m_outer = mu * m_outer + g
        x_t = x_t + alpha * m_outer.sign()

        x_t = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
        # Make sure we are still in bounds
        x_t = torch.clamp(x_t, 0., 1.)

    return x_t.clone().detach()


class DataTransform(nn.Module):
    """
    Generic class for block-wise transformation.
    """

    def __init__(self ):
        super().__init__()
        self.block_size = 4
        self.blocks_axis0 = int(32 / 4)
        self.blocks_axis1 = int(32 /4)
        self.mean = torch.Tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.Tensor([0.2471, 0.2435, 0.2616])

    def segment(self, X):
        X = X.permute(0, 2, 3, 1)
        X = X.reshape(
            -1,
            self.blocks_axis0,
            self.block_size,
            self.blocks_axis1,
            self.block_size,
            3,
        )
        X = X.permute(0, 1, 3, 2, 4, 5)
        X = X.reshape(
            -1,
            self.blocks_axis0,
            self.blocks_axis1,
            self.block_size * self.block_size * 3,
        )
        return X

    def integrate(self, X):
        X = X.reshape(
            -1,
            self.blocks_axis0,
            self.blocks_axis1,
            self.block_size,
            self.block_size,
            3,
        )
        X = X.permute(0, 1, 3, 2, 4, 5)
        X = X.reshape(
            -1,
            self.blocks_axis0 * self.block_size,
            self.blocks_axis1 * self.block_size,
            3,
        )
        X = X.permute(0, 3, 1, 2)
        return X

    def generate_key(self, seed, binary=False):
        torch.manual_seed(seed)
        key = torch.randperm(self.block_size * self.block_size * 3)
        if binary:
            key = key > len(key) / 2
        return key

    def normalize(self, X):
        return (X - self.mean.type_as(X)[None, :, None, None]) / self.std.type_as(X)[
            None, :, None, None
        ]

    def denormalize(self, X):
        return (X * self.std.type_as(X)[None, :, None, None]) + self.mean.type_as(X)[
            None, :, None, None
        ]

    def forward(self, X, decrypt=False):
        raise NotImplementedError
def mask(size, alpha):
    # size = original.shape
    # print(size)
    mask_innner = torch.zeros(size=size).cuda()
    mask_outer = torch.zeros(size=size).cuda()

    for x in range(size[0]):
        for y in range(size[1]):
            for z in range(size[2]):
                if y < mask_innner.size()[1] * alpha or y >= mask_innner.size()[1]*(1-alpha) or z < mask_innner.size()[2] * alpha or z >= mask_innner.size()[2]*(1-alpha) :
                    mask_outer[x][y][z] = 1
                else:
                    mask_innner[x][y][z] = 1

    return mask_innner , mask_outer

class HighFrequenceMask(DataTransform):
    def __init__(self,seed):
        super().__init__()
        # np.random.seed(seed)
        size = [3,32,32]
        self.fbound = 0 # 之前1.69
        # inner_mask 就是里面有值，外面全零；outer_mask就是外面为1，里面全零
        self.inner_mask, self.outer_mask=mask(size, alpha = 1/4)
        self.err_rate_path = '/data/zwl/zwl/DVERGE_code/eval/err_c100.pt'
        self.err_rate_bound = 0.4183
    def get_fbmask(self, X):
    # size = original.shape
    # print(size)
        size = X.shape
        mask_sens = torch.zeros(size=size).cuda()
        # mask_outer = torch.zeros(size=size).cuda()
        img_c3 = torch.fft.fft2(X)
        mask_sens = torch.where(torch.absolute(img_c3) > self.fbound , 1 , 0)

        return mask_sens

    def get_errate_mask(self):
        # size = original.shape
        err_rate = torch.load(self.err_rate_path)
        mask = torch.where(err_rate > self.err_rate_bound,1 , 0 ) 
        return mask.cuda()
      
    def forward(self, X):
        size = X.shape
        noise = torch.randint(255, size) / 255
        fft_noise = torch.fft.fft2(noise).cuda()
        # fft_noise = torch.fft.fftshift(fft_noise)
        img_c3 = torch.fft.fft2(X)
        # img_c3 = torch.fft.fftshift(img_c3)
        mask = self.get_errate_mask()
        # mask 丢掉低频, 可以保证攻击的性能
        # img_c3 = img_c3 * self.outer_mask
        # img_c3 = torch.where(torch.absolute(img_c3) > self.fbound , img_c3 , fft_noise)
        # mask 丢掉高频, 3/8不能保证攻击的性能, 1/4 的攻击性能大概降到0.5
        # img_c3 =  torch.where(mask == 0 , img_c3 , fft_noise)
        img_c3 = img_c3 * mask
        ################ 该方法相当于对所有扰动过滤低频信息  ##########
        # mask_img = img_c3 * self.outer_mask
        # mask_noise = fft_noise * self.inner_mask
        # mask_noise = 0 * self.inner_mask
        # mask_noise = x_nat * self.inner_mask

        # img_c3 = mask_img + mask_noise
        img_c4 = torch.fft.ifft2(img_c3)
        # img_c4 = torch.fft.ifft2(torch.fft.ifftshift(img_c3))
        return torch.real(img_c4)

class HighFrequenceEdit(DataTransform):
    def __init__(self,fb):
        super().__init__()
        # np.random.seed(seed)
        size = [3,32,32]
        self.fbound = fb # 之前1.69
        # inner_mask 就是里面有值，外面全零；outer_mask就是外面为1，里面全零
        # self.inner_mask, self.outer_mask=mask(size, alpha = 1/4)
        # self.err_rate_path = '/data/zwl/zwl/DVERGE_code/eval/err_c100.pt'
        # self.err_rate_bound = 0.4163
    def get_fbmask(self, X):
    # size = original.shape
    # print(size)
        size = X.shape
        mask_sens = torch.zeros(size=size).cuda()
        # mask_outer = torch.zeros(size=size).cuda()
        img_c3 = torch.fft.fft2(X)
        mask_sens = torch.where(torch.absolute(img_c3) > self.fbound , 1 , 0)

        return mask_sens
 
    # def get_errate_mask(self):
    #     # size = original.shape
    #     err_rate = torch.load(self.err_rate_path)
    #     mask = torch.where(err_rate > self.err_rate_bound,1 , 0 ) 
    #     return mask.cuda()
      
    def forward(self, X,G):
        size = X.shape
        noise = torch.randint(255, size) / 255
        noise_fft = torch.fft.fft2(noise).cuda()
        G_fft = torch.fft.fft2(G).cuda()
        # fft_noise = torch.fft.fftshift(fft_noise)
        img_c3 = torch.fft.fft2(X)
        # img_c3 = torch.fft.fftshift(img_c3)
        mask = self.get_fbmask(X)
        # mask 丢掉低频, 可以保证攻击的性能
        # img_c3 = img_c3 * self.outer_mask
        # img_c3 = torch.where(torch.absolute(img_c3) > self.fbound , img_c3 , fft_noise)
        # mask 丢掉高频, 3/8不能保证攻击的性能, 1/4 的攻击性能大概降到0.5
        img_c3 =  torch.where(mask == 1 ,  G_fft, noise_fft)
        # img_c3 = img_c3 * mask
        ################ 该方法相当于对所有扰动过滤低频信息  ##########
        # mask_img = img_c3 * self.outer_mask
        # mask_noise = fft_noise * self.inner_mask
        # mask_noise = 0 * self.inner_mask
        # mask_noise = x_nat * self.inner_mask

        # img_c3 = mask_img + mask_noise
        img_c4 = torch.fft.ifft2(img_c3)
        # img_c4 = torch.fft.ifft2(torch.fft.ifftshift(img_c3))
        return torch.abs(img_c4)
