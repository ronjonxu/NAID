import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
import torch
from util.util import torch_save
import math 
import torch.nn.functional as F
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BaseModel(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.scale = opt.scale

        if len(self.gpu_ids) > 0:
            self.device = torch.device('cuda', self.gpu_ids[0])
        else:
            self.device = torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.optimizer_names = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.start_epoch = 0
                
        self.backwarp_tenGrid = {}
        self.backwarp_tenPartial = {}

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def setup(self, opt=None):
        opt = opt if opt is not None else self.opt
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) \
                               for optimizer in self.optimizers]
            for scheduler in self.schedulers:
                scheduler.last_epoch = opt.load_iter
        if opt.load_iter > 0 or opt.load_path != '':
            load_suffix = opt.load_iter
            self.load_networks(load_suffix)
            if opt.load_optimizers:
                self.load_optimizers(opt.load_iter)

        self.print_networks(opt.verbose)

    def eval(self):
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            net.eval()

    def train(self):
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            net.train()

    def test(self):
        with torch.no_grad():
            self.forward()

    def get_image_paths(self):
        return self.image_paths

    def update_learning_rate(self):
        for i, scheduler in enumerate(self.schedulers):
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
            print('lr of %s = %.7f' % (
                    self.optimizer_names[i], scheduler.get_last_lr()[0]))

    def update_learning_rate_cosine(self):
        for i, scheduler in enumerate(self.schedulers):
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
                
    def print_learning_rate_cosine(self):
        for i, scheduler in enumerate(self.schedulers):
            print('lr of %s = %.7f' % (
                    self.optimizer_names[i], scheduler.get_last_lr()[0]))
                    
    def print_learning_rate_cosine_setting(self):
        for i, scheduler in enumerate(self.schedulers):
            print('last_epoch: %d' % (scheduler.last_epoch))

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(getattr(self, name), (list, tuple)):
                visual_ret[name] = torch.clamp(
                        getattr(self, name)[0].detach()*255, 0, 255).round()
            else:	
                visual_ret[name] = torch.clamp(
                        getattr(self, name).detach()*255, 0, 255).round()
                if 'res' in name:
                    print(torch.max(visual_ret[name]))

        return visual_ret
    
    def get_current_visuals_dnd(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(getattr(self, name), (list, tuple)):
                visual_ret[name] = torch.clamp(
                        getattr(self, name)[0].detach()*255, 0, 255).round()
            else:	
                visual_ret[name] = torch.clamp(
                        getattr(self, name).detach()*255, 0, 255).round()
                if 'res' in name:
                    print(torch.max(visual_ret[name]))

        return visual_ret

    def get_current_visuals_dvd(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(getattr(self, name), (list, tuple)):
                visual_ret[name] = getattr(self, name)[0].detach()
            else:	
                visual_ret[name] = getattr(self, name).detach()

        return visual_ret



    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):
        for name in self.model_names:
            save_filename = '%s_model_%d.pth' % (name, epoch)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, 'net' + name)
            if self.device.type == 'cuda':
                state = {'state_dict': net.module.cpu().state_dict()}
                torch_save(state, save_path)
                net.to(self.device)
            else:
                state = {'state_dict': net.state_dict()}
                torch_save(state, save_path)
        self.save_optimizers(epoch)

    def load_networks(self, epoch):
        for name in self.model_names: #[0:1]:
            print(name)
            load_filename = '%s_model_%d.pth' % (name, epoch)
            if self.opt.load_path != '':
                load_path = self.opt.load_path
            else:
                load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            state_dict = torch.load(load_path, map_location=self.device)
            print('loading the model from %s' % (load_path))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            net_state = net.state_dict()
            is_loaded = {n:False for n in net_state.keys()}
            for name, param in state_dict['state_dict'].items():
            # for name, param in state_dict['params'].items():
                # load dvn_fusion
                name = name.replace('module.', '')
                # print(name)
                if name in net_state:
                    try:
                        net_state[name].copy_(param)
                        is_loaded[name] = True
                    except Exception:
                        print('While copying the parameter named [%s], '
                              'whose dimensions in the model are %s and '
                              'whose dimensions in the checkpoint are %s.'
                              % (name, list(net_state[name].shape),
                                 list(param.shape)))
                        raise RuntimeError
                else:
                    print('Saved parameter named [%s] is skipped' % name)
            mark = True
            for name in is_loaded:
                if not is_loaded[name]:
                    print('Parameter named [%s] is randomly initialized' % name)
                    mark = False
                    # raise RuntimeError
            if mark:
                print('All parameters are initialized using [%s]' % load_path)

            self.start_epoch = epoch

    def load_network_path(self, net, path):
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        state_dict = torch.load(path, map_location=self.device)
        print('loading the model from %s' % (path))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        net_state = net.state_dict()
        is_loaded = {n:False for n in net_state.keys()}
        if 'state_dict' in state_dict.keys() or 'params' in state_dict.keys():
            param_keys = 'state_dict' if 'state_dict' in state_dict.keys() else 'params'
            for name, param in state_dict[param_keys].items():
            # for name, param in state_dict['params'].items():
                if name in net_state:
                    try:
                        net_state[name].copy_(param)
                        is_loaded[name] = True
                    except Exception:
                        print('While copying the parameter named [%s], '
                                'whose dimensions in the model are %s and '
                                'whose dimensions in the checkpoint are %s.'
                                % (name, list(net_state[name].shape),
                                    list(param.shape)))
                        raise RuntimeError
                else:
                    print('Saved parameter named [%s] is skipped' % name)
        else:
            for name, param in state_dict.items():
                name = name.replace('module.', '')
                if name in net_state:
                    try:
                        net_state[name].copy_(param)
                        is_loaded[name] = True
                    except Exception:
                        print('While copying the parameter named [%s], '
                                'whose dimensions in the model are %s and '
                                'whose dimensions in the checkpoint are %s.'
                                % (name, list(net_state[name].shape),
                                    list(param.shape)))
                        raise RuntimeError
                else:
                    print('Saved parameter named [%s] is skipped' % name)
        mark = True
        for name in is_loaded:
            if not is_loaded[name]:
                print('Parameter named [%s] is randomly initialized' % name)
                mark = False
        if mark:
            print('All parameters are initialized using [%s]' % path)


    def load_network_path_naf(self, net, path):
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        state_dict = torch.load(path, map_location=self.device)
        print('loading the model from %s' % (path))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        net_state = net.state_dict()
        is_loaded = {n:False for n in net_state.keys()}
        # print(state_dict['params'].keys())
        for name, param in state_dict['state_dict'].items():
        # for name, param in state_dict['params'].items():
            name = name.replace('NAF', 'NAFF')
            if name in net_state:
                try:
                    net_state[name].copy_(param)
                    is_loaded[name] = True
                except Exception:
                    print('While copying the parameter named [%s], '
                            'whose dimensions in the model are %s and '
                            'whose dimensions in the checkpoint are %s.'
                            % (name, list(net_state[name].shape),
                                list(param.shape)))
                    raise RuntimeError
            else:
                print('Saved parameter named [%s] is skipped' % name)
        mark = True
        for name in is_loaded:
            if not is_loaded[name]:
                print('Parameter named [%s] is randomly initialized' % name)
                mark = False
        if mark:
            print('All parameters are initialized using [%s]' % path)

    def load_network_path_dvn(self, net, path):
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        state_dict = torch.load(path, map_location=self.device)
        print('loading the model from %s' % (path))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        net_state = net.state_dict()
        is_loaded = {n:False for n in net_state.keys()}
        # print(state_dict['params'].keys())
        for name, param in state_dict['state_dict'].items():
        # for name, param in state_dict['params'].items():
            name = name.replace('module.', '')
            if name in net_state:
                try:
                    net_state[name].copy_(param)
                    is_loaded[name] = True
                except Exception:
                    print('While copying the parameter named [%s], '
                            'whose dimensions in the model are %s and '
                            'whose dimensions in the checkpoint are %s.'
                            % (name, list(net_state[name].shape),
                                list(param.shape)))
                    raise RuntimeError
            else:
                print('Saved parameter named [%s] is skipped' % name)
        mark = True
        for name in is_loaded:
            if not is_loaded[name]:
                print('Parameter named [%s] is randomly initialized' % name)
                mark = False
        if mark:
            print('All parameters are initialized using [%s]' % path)

    def save_optimizers(self, epoch):
        assert len(self.optimizers) == len(self.optimizer_names)
        for id, optimizer in enumerate(self.optimizers):
            save_filename = self.optimizer_names[id]
            state = {'name': save_filename,
                     'epoch': epoch,
                     'state_dict': optimizer.state_dict()}
            save_path = os.path.join(self.save_dir, save_filename+'.pth')
            torch_save(state, save_path)

    def load_optimizers(self, epoch):
        assert len(self.optimizers) == len(self.optimizer_names)
        for id, optimizer in enumerate(self.optimizer_names):
            load_filename = self.optimizer_names[id]
            load_path = os.path.join(self.save_dir, load_filename+'.pth')
            print('loading the optimizer from %s' % load_path)
            state_dict = torch.load(load_path)
            assert optimizer == state_dict['name']
            assert epoch == state_dict['epoch']
            self.optimizers[id].load_state_dict(state_dict['state_dict'])

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M'
                      % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
