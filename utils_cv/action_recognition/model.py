# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import OrderedDict
import os
import warnings
import numpy as np
from typing import Any, Callable, Dict, List, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import torch.cuda as cuda
from sklearn.metrics import accuracy_score

try:
    from apex import amp

    AMP_AVAILABLE = True
except ModuleNotFoundError:
    AMP_AVAILABLE = False

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# this
from collections import deque
import io
import decord
import IPython.display
from time import sleep, time
from PIL import Image
from threading import Thread
from torchvision.transforms import Compose
import torchvision.models.video as torch_video_models
from utils_cv.action_recognition.dataset import get_transforms

from ..common.gpu import torch_device, num_devices
from .dataset import VideoDataset, DEFAULT_STD, DEFAULT_MEAN

from .references.metrics import accuracy, AverageMeter
from .references import functional_video as F

# These paramaters are set so that we can use torch hub to download pretrained
# models from the specified repo
TORCH_R2PLUS1D = "moabitcoin/ig65m-pytorch"  # From https://github.com/moabitcoin/ig65m-pytorch
MODELS = {
    # Model name followed by the number of output classes.
    "r2plus1d_34_32_ig65m": 359,
    "r2plus1d_34_32_kinetics": 400,
    "r2plus1d_34_8_ig65m": 487,
    "r2plus1d_34_8_kinetics": 400,
    "mc3_18": 400,
    "r2plus1d_18": 400,
    "r3d_18": 400,
}

class Perturbation(nn.Module):
    def __init__(self, size,requires_grad=True,
                 device='cuda',
                 max_value=None,
                 min_value=None,
                 max_norm=1.0,
                 cyclic_pert=False):
        super(Perturbation, self).__init__()
        
        self.size = size
        self.device = device
        self.requires_grad =requires_grad
        # self.perturbation =  torch.nn.Parameter(torch.zeros(size=size,requires_grad=requires_grad,device=device))
        self.perturbation = torch.nn.Parameter(torch.rand(size=self.size, requires_grad=requires_grad, device=device).mul(2).sub(1).mul(0.000001))
        if max_value is None:
            self.max_value=np.min((1-np.array(DEFAULT_MEAN))/DEFAULT_STD)
        if min_value is None:
            self.min_value=np.max((0.-np.array(DEFAULT_MEAN))/DEFAULT_STD)
        self.max_norm = max_norm
        self.dynamic_max_norm =max_norm
        self.cyclic_pert=cyclic_pert

    def forward(self, *input):
        x,adversarial = input[0]
        if not adversarial:
            return x
        
        # with torch.no_grad():
        #     self.perturbation.clamp_(-1.*self.max_norm,self.max_norm)
        # self.perturbation = self.max_norm*torch.tanh(self.perturbation)
        self.perturbation_clamp = self.clamp_perturbation()
        self.perturbation_normalize=F.normalize(self.perturbation_clamp, mean=(0,0,0), std=DEFAULT_STD, inplace=False)
        
        if self.cyclic_pert:
            self.perturbation_normalize =torch.roll(self.perturbation_normalize,shifts=np.random.randint(0, self.size[1]),dims=1)
        
        self.adversdarial_input = x+self.perturbation_normalize
        self.adversdarial_input_clip = self.adversdarial_input.clamp(self.min_value,self.max_value)
        return self.adversdarial_input_clip

    def clamp_perturbation(self):
        # self.perturbation_clamp = self.perturbation.clamp(-1.*self.max_norm,self.max_norm)
        self.dynamic_max_norm * self.perturbation.tanh()
        return self.perturbation.clamp(-1.*self.dynamic_max_norm,self.dynamic_max_norm)

    def convert_adversarial_video_zero_one(self, adv_vid):
        x_hat_np = adv_vid.data.cpu().numpy()
        x_hat_np_norm = (x_hat_np.transpose([0, 2, 3, 4, 1]) + np.array(DEFAULT_MEAN) / np.array(DEFAULT_STD)) * np.array(
            DEFAULT_STD)
        return x_hat_np_norm

    def apply_perturbation(self, x):
        x_hat  = self.forward([x, True])
        x_hat_np_norm =self.convert_adversarial_video_zero_one(x_hat)
        return  x_hat_np_norm

    def metric_calc(self):

        thickness = self.perturbation.abs().mean()*100.
        roughness = (torch.roll(self.perturbation, 1, dims=1)-self.perturbation).abs().mean()*100.

        return thickness, roughness
    
    def init_perturbation(self, perturbation=[],requires_grad=True,device='cuda'):
        if len(perturbation)==0:
            self.perturbation = torch.nn.Parameter(
                torch.rand(size=self.size, requires_grad=self.requires_grad, device=self.device).mul(2).sub(1).mul(0.000001))
        else:
            self.perturbation =  torch.nn.Parameter(torch.from_numpy(perturbation))

    def get_perturbation(self):
        return self.clamp_perturbation(), self.perturbation

class Losses():
    def __init__(self, beta_1=0.5,
                 lambda_=1.0,
                 targeted=False,
                 target_class=None,
                 margin=0.05,
                 improve_loss=False,
                 logits=False,
                 attack_type="flickering"):
        super(Losses).__init__()

        self.beta_1=beta_1
        self.lambda_ = lambda_
        self.targeted= targeted
        self.target_class =target_class
        self.margin= torch.FloatTensor([margin]).to('cuda')
        self.logits =logits
        self.attack_type= attack_type

        if improve_loss:
            self.adv_loss = self.improve_adversarial_loss
            if self.logits:
                print(f"Loss: using improve loss with logits")
            else:
                print(f"Loss: using improve loss without logits (prob)")
                
        else:
            self.adv_loss = self.ce_adversarial_loss
            print(f"Loss: using ce loss")
            
        if self.attack_type == "flickering":
            self.regularization_loss = self.flickering_regularization_loss
            print(f"Attack type: {self.attack_type}, using flickering_regularization_loss")
        else:
            self.regularization_loss = self.L12_regularization_loss
            print(f"Attack type: {self.attack_type}, using L12_regularization_loss")   


    def __call__(self, labels, model_logits, model_prob, perturbation):

        reg_loss =  self.regularization_loss(perturbation)
        adv_loss = self.adv_loss(labels, model_logits, model_prob)
        loss = adv_loss + self.lambda_ * reg_loss

        return [loss, adv_loss, reg_loss]

    def ce_adversarial_loss(self, labels, model_logits, model_prob):
        if self.targeted:
            # self.ce_adversarial_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
            #                                                                           logits=self.model_logits)
            # self.to_min_prob = self.max_non_label_prob
            # self.to_max_prob = self.label_prob
            label_prob = model_prob[:,self.target_class]
            ce_adversarial_loss = -torch.log(label_prob + 1e-6)
            self.label_prob =  model_prob[:,self.target_class]
        else:
            # self.ce_adversarial_loss = -1.0*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.model_logits)
            # self.to_min_prob = self.label_prob
            # self.to_max_prob = self.max_non_label_prob
            self.label_prob = model_prob.gather(1, labels.view(-1,1))
            ce_adversarial_loss = -torch.log(1 - self.label_prob + 1e-6)
            # self.ce_adversarial_loss = -1.0*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.model_logits)
            # self.to_min_prob = self.label_prob
            # self.to_max_prob = self.max_non_label_prob

        return torch.mean(ce_adversarial_loss)

    def flickering_regularization_loss(self, perturbation):

        norm_reg = torch.mean(perturbation ** 2) + 1e-12

        perturbation_roll_right = torch.roll(perturbation, 1, dims=1)
        perturbation_roll_left = torch.roll(perturbation, -1, dims=1)

        diff_norm_reg = torch.mean((perturbation - perturbation_roll_right) ** 2) + 1e-12
        laplacian_norm_reg = torch.mean((-2 * perturbation + perturbation_roll_right + perturbation_roll_left) ** 2) + 1e-12

        reg_loss = self.beta_1*norm_reg + (1-self.beta_1)*(diff_norm_reg + laplacian_norm_reg)
        return reg_loss
    
    def L12_regularization_loss(self, perturbation):
        reg_loss = torch.sum(torch.sqrt(torch.mean(perturbation**2,[0,2,3])))+1e-12
        
        return reg_loss

    def improve_adversarial_loss(self,  labels, model_logits, model_prob):
        self.label_prob = model_prob.gather(1, labels.view(-1, 1))
        self.idx_non_label = (1. - torch.nn.functional.one_hot(labels, 400)).type(torch.BoolTensor)
        self.max_non_label_prob =model_prob[self.idx_non_label].reshape([model_prob.shape[0],-1]).max(dim=1)[0].reshape([model_prob.shape[0],1])

        if self.targeted:
            if self.logits:
                to_min_elem = self.max_non_label_logits
                to_max_elem = self.label_logits
                loss_margin = torch.log(1. + self.margin * (1. / label_prob))
            else:
                to_min_elem = self.max_non_label_prob
                to_max_elem = self.label_prob
                loss_margin = self.margin

        else:
            if self.logits:
                self.label_logits = model_logits.gather(1, labels.view(-1, 1))
                self.to_min_elem = self.label_logits
                self.to_max_elem = model_logits[self.idx_non_label].reshape([model_logits.shape[0],-1]).max(dim=1)[0].reshape(self.to_min_elem.shape)
                self.loss_margin = torch.log(1. + self.margin * (1. / (0.00001 + self.label_prob)))
            else:
                self.to_min_elem = self.label_prob
                self.to_max_elem = self.max_non_label_prob
                self.loss_margin = self.margin

        self.l_1 = torch.zeros_like(self.to_min_elem)
        self.l_2 = ((self.to_min_elem - (self.to_max_elem - self.loss_margin)) ** 2) / self.loss_margin
        self.l_3 = self.to_min_elem - (self.to_max_elem - self.loss_margin)


        adversarial_loss = torch.max(self.l_1, torch.min(self.l_2,self.l_3))
        adversarial_loss_total = torch.sum(adversarial_loss)

        return adversarial_loss_total


class Adversarial_metrics():
    """Computes the accuracy over the k top predictions for the specified values of k"""

    def __init__(self, targeted=False, target_class=None):
        super(Adversarial_metrics).__init__()

        self.targeted= targeted
        self.target_class =target_class

    def accuracy(self,output, ground_truth, topk=(1,), clean_pred=None):

        with torch.no_grad():
            res = []
            batch_size = ground_truth.size(0)

            if self.targeted:
                maxk = max(topk)
                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(self.target_class)
                correct =correct.view(-1).float().sum(0, keepdim=True)
                res.append(correct[0].mul_(100.0 / batch_size))
            else:
                maxk = max(topk)

                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(ground_truth.view(1, -1).expand_as(pred))

                _, pred_no_adv = clean_pred.topk(maxk, 1, True, True)
                pred_no_adv = pred_no_adv.t()
                correct_no_adv = pred_no_adv.eq(ground_truth.view(1, -1).expand_as(pred_no_adv))

                for k in topk:
                    correct_k = (correct[:k] * correct_no_adv[:k]).view(-1).float().sum(0, keepdim=True)
                    # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                    res.append((1 - correct_k.mul(1. / correct_no_adv[:k].view(-1).float().sum())).mul(100.))

            return res
        
    def accuracy_for_eval(self,output, ground_truth, topk=(1,), clean_pred=None):

        with torch.no_grad():
            res = []
            batch_size = ground_truth.size(0)
        
            if self.targeted:
                maxk = max(topk)
                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(self.target_class)
                correct =correct.view(-1).float().sum(0, keepdim=True)
                res.append(correct[0].mul_(100.0 / batch_size))
            else:
                maxk = max(topk)
        
                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(ground_truth.view(1, -1).expand_as(pred))
        
                _, pred_no_adv = clean_pred.topk(maxk, 1, True, True)
                pred_no_adv = pred_no_adv.t()
                correct_no_adv = pred_no_adv.eq(ground_truth.view(1, -1).expand_as(pred_no_adv))

                for k in topk:
                    miss = (torch.logical_not(correct[:k]) * correct_no_adv[:k]).view(-1).float().sum(0, keepdim=True)
                    # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                    # res.append((1 - correct_k.mul(1. / correct_no_adv[:k].view(-1).float().sum())).mul(100.))
                    num_correct_no_adv =correct_no_adv[:k].view(-1).float().sum()
        
            return miss, num_correct_no_adv

    def adversarial_metric(self,perturbation):

        thickness = perturbation.abs().mean()*100.
        roughness = (torch.roll(perturbation, 1, dims=1)-perturbation).abs().mean()*100.

        return thickness, roughness



class VideoLearnerAdversarial(object):
    """ Video recognition learner object that handles training loop and evaluation. """

    def __init__(
        self,
        dataset: VideoDataset = None,
        num_classes: int = None,  # ie 51 for hmdb51
        base_model: str = "ig65m",  # or "kinetics"
        sample_length: int = None,
        cyclic_pert: bool = False,
        l_inf_pert_norm: float=1.,
        attack_type: str ="flickering",
        labaels_id_to_text =None
    ) -> None:
        """ By default, the Video Learner will use a R2plus1D model. Pass in
        a dataset of type Video Dataset and the Video Learner will intialize
        the model.

        Args:
            dataset: the datset to use for this model
            num_class: the number of actions/classifications
            base_model: the R2plus1D model is based on either ig65m or
            kinetics. By default it will use the weights from ig65m since it
            tends attain higher results.
        """
        # set empty - populated when fit is called
        self.results = []

        # set num classes
        self.num_classes = num_classes
        
        self.attack_type = attack_type
        
        self.label_id_to_text = labaels_id_to_text
        
        if dataset:
            self.dataset = dataset
            self.sample_length = self.dataset.sample_length
        else:
            assert sample_length == 8 or sample_length == 32
            self.sample_length = sample_length

        self.model, self.model_name, self.num_classes = self.init_model(
            self.sample_length, base_model, num_classes,
        )
        
        if self.attack_type =="flickering":
            pert_size= torch.Size([3,self.sample_length,1,1])
            
        else:
            pert_size= torch.Size([3,self.sample_length,112,112])
            
        print(f"Attack type: {self.attack_type}, perturbation shape: {pert_size}")  
        
        self.pert_model=Perturbation(size=pert_size,
                                     requires_grad=True,
                                     device=torch_device(),
                                     max_norm=l_inf_pert_norm,
                                     cyclic_pert=cyclic_pert)
        
        self.model.eval()
        self.r2plus1_model = self.model
        
        self.model = torch.nn.Sequential(self.pert_model,self.r2plus1_model)
        self.model.eval()
        self._set_requires_grad(True)
        

    @staticmethod
    def init_model(
        sample_length: int, base_model: str, num_classes: int = None
    ) -> torchvision.models.video.resnet.VideoResNet:
        """
        Initializes the model by loading it using torch's `hub.load`
        functionality. Uses the model from TORCH_R2PLUS1D.

        Args:
            sample_length: Number of consecutive frames to sample from a video (i.e. clip length).
            base_model: the R2plus1D model is based on either ig65m or kinetics.
            num_classes: the number of classes/actions

        Returns:
            Load a model from a github repo, with pretrained weights
        """
        if base_model not in ("ig65m", "kinetics"):
            model_name =base_model
            print(f"Loading {base_model} model")
            model = getattr(torch_video_models,base_model)(True,True)
        else:
            model_name = f"r2plus1d_34_{sample_length}_{base_model}"

            print(f"Loading {model_name} model")

            model = torch.hub.load(
                TORCH_R2PLUS1D,
                model_name,
                num_classes=MODELS[model_name],
                pretrained=True,
            )


        # Replace head
        if num_classes is not None:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            num_of_classes=MODELS[model_name]

        return model, model_name, num_of_classes

    def freeze(self) -> None:
        """Freeze model except the last layer"""
        self._set_requires_grad(False)
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze(self) -> None:
        """Unfreeze all layers in model"""
        self._set_requires_grad(True)

    def _set_requires_grad(self, requires_grad=True) -> None:
        """ sets requires grad """
        for param in self.model.parameters():
            param.requires_grad = requires_grad



    def fit(
        self,
        lr: float,
        epochs: int,
        model_dir: str = "checkpoints",
        model_name: str = None,
        momentum: float = 0.95,
        weight_decay: float = 0.0001,
        mixed_prec: bool = False,
        use_one_cycle_policy: bool = False,
        warmup_pct: float = 0.3,
        lr_gamma: float = 0.1,
        lr_step_size: float = None,
        grad_steps: int = 2,
        save_model: bool = False,
        loss_params_dict= None,
        devices_ids =None,
        start_epoch:int =1
    ) -> None:
        """ The primary fit function """
        # set epochs
        self.epochs = epochs

        #set losses
        self.loss_handler =Losses(beta_1=loss_params_dict['beta_1'],
                                  lambda_=loss_params_dict['lambda_'],
                                  targeted=loss_params_dict['targeted_attack'],
                                  target_class=loss_params_dict['target_class_id'],
                                  improve_loss=loss_params_dict['improve_loss'],
                                  logits=loss_params_dict['use_logits'],
                                  attack_type=self.attack_type)

        self.metric_handler = Adversarial_metrics(targeted=loss_params_dict['targeted_attack'],
                                                  target_class=loss_params_dict['target_class_id'])

        # set lr_step_size based on epochs
        if lr_step_size is None:
            lr_step_size = np.ceil(2 / 3 * self.epochs)

        # set model name
        if model_name is None:
            model_name = self.model_name

        os.makedirs(model_dir, exist_ok=True)

        data_loaders = {}
        data_loaders["train"] = self.dataset.train_dl
        data_loaders["valid"] = self.dataset.test_dl

        # Move model to gpu before constructing optimizers and amp.initialize
        device = torch_device()
        self.model.to(device)
        if devices_ids:
            count_devices = len(devices_ids)
        else:
            count_devices = num_devices()
            devices_ids = list(range(count_devices))
        
        torch.backends.cudnn.benchmark = True

        named_params_to_update = {}
        total_params = 0
        for name, param in self.pert_model.named_parameters():
            total_params += 1
            if param.requires_grad:
                named_params_to_update[name] = param

        print("Params to learn:")
        if len(named_params_to_update) == total_params:
            print("\tfull network")
        else:
            for name in named_params_to_update:
                print(f"\t{name}")

        # create optimizer
        # optimizer = optim.SGD(
        #     list(named_params_to_update.values()),
        #     lr=lr,
        #     momentum=momentum,
        #     weight_decay=weight_decay,
        # )

        optimizer = torch.optim.Adam(list(named_params_to_update.values()), lr=lr)

        # Use mixed-precision if available
        # Currently, only O1 works with DataParallel: See issues https://github.com/NVIDIA/apex/issues/227
        if mixed_prec:
            # break if not AMP_AVAILABLE
            assert AMP_AVAILABLE
            # 'O0': Full FP32, 'O1': Conservative, 'O2': Standard, 'O3': Full FP16
            self.model, optimizer = amp.initialize(
                self.model,
                optimizer,
                opt_level="O1",
                loss_scale="dynamic",
                # keep_batchnorm_fp32=True doesn't work on 'O1'
            )

        # Learning rate scheduler
        if use_one_cycle_policy:
            # Use warmup with the one-cycle policy
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=self.epochs,
                pct_start=warmup_pct,
                base_momentum=0.9 * momentum,
                max_momentum=momentum,
            )
        else:
            # Simple step-decay
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=lr_step_size, gamma=lr_gamma,
            )

        # DataParallel after amp.initialize
        model = (
            nn.DataParallel(self.model,device_ids=devices_ids) if count_devices > 1 else self.model
        )

        # criterion = nn.CrossEntropyLoss().to(device)

        # set num classes
        topk = 5
        if topk >= self.num_classes:
            topk = self.num_classes

        for e in range(start_epoch, self.epochs + 1):
            print(
                f"Epoch {e} ========================================================="
            )
            print(f"lr={scheduler.get_lr()}")

            self.results.append(
                self.train_an_epoch(
                    model,
                    self.pert_model,
                    data_loaders,
                    device,
                    self.loss_handler,
                    self.metric_handler,
                    optimizer,
                    grad_steps=grad_steps,
                    mixed_prec=mixed_prec,
                    topk=topk,
                )
            )

            scheduler.step()
            os.makedirs(model_dir, exist_ok=True)
            if save_model:
                # self.save(
                #     os.path.join(
                #         model_dir,
                #         "{model_name}_{epoch}.pt".format(
                #             model_name=model_name, epoch=str(e).zfill(3),
                #         ),
                #     )
                # )
                np.save(os.path.join(
                        model_dir,
                        "{model_name}_{epoch}.npy".format(
                            model_name=model_name, epoch=str(e).zfill(3),
                        )),self.results)
        # self.plot_precision_loss_curves()

    @staticmethod
    def train_an_epoch(
        model,
        pert_model,
        data_loaders,
        device,
        criterion,
        metric,
        optimizer,
        grad_steps: int = 1,
        mixed_prec: bool = False,
        topk: int = 5,
    ) -> Dict[str, Any]:
        """Train / validate a model for one epoch.

        Args:
            model: the model to use to train
            data_loaders: dict {'train': train_dl, 'valid': valid_dl}
            device: gpu or not
            criterion: TODO
            optimizer: TODO
            grad_steps: If > 1, use gradient accumulation. Useful for larger batching
            mixed_prec: If True, use FP16 + FP32 mixed precision via NVIDIA apex.amp
            topk: top k classes

        Return:
            dict {
                'train/time': batch_time.avg,
                'train/loss': losses.avg,
                'train/top1': top1.avg,
                'train/top5': top5.avg,
                'valid/time': ...
            }
        """
        if mixed_prec and not AMP_AVAILABLE:
            warnings.warn(
                """
                NVIDIA apex module is not installed. Cannot use
                mixed-precision. Turning off mixed-precision.
                """
            )
            mixed_prec = False

        result = OrderedDict()
        for phase in ["train", "valid"]:
            # switch mode
            # if phase == "train":
            #     model.train()
            # else:
            #     model.eval()

            # set loader
            dl = data_loaders[phase]

            # collect metrics
            batch_time = AverageMeter()
            losses = AverageMeter()
            loss_adv_l = AverageMeter()
            loss_reg_l = AverageMeter()
            fooling_ratio_l = AverageMeter()
            top5 = AverageMeter()
            miss_rate=0
            total_val_vid=0

            end = time()
            for step, (inputs, target, _) in enumerate(dl, start=0):
                if step % 10 == 0:
                    print(f" Phase {phase}: batch {step} of {len(dl)}")
                inputs = inputs.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                with torch.no_grad():
                    outputs_no_adv = model([inputs,False])
                    scores_no_adv = nn.functional.softmax(outputs_no_adv, dim=1)
                    
                with torch.set_grad_enabled(phase == "train"):
                    # compute output
                    outputs = model([inputs, True])

                    scores = nn.functional.softmax(outputs, dim=1)
                    perturbation = pert_model.get_perturbation()[0]
                    loss, adv_loss, reg_loss = criterion(target,outputs,scores,perturbation)

                    # measure accuracy and record loss

                    fooling, valid_vid  = metric.accuracy_for_eval(outputs, target, topk=(1,), clean_pred=outputs_no_adv)
                    miss_rate += fooling[0].data.cpu().numpy()
                    total_val_vid += valid_vid.data.cpu().numpy()

                    losses.update(loss.item(), inputs.size(0))
                    # fooling_ratio_l.update(fooling_ratio[0][0], inputs.size(0))
                    loss_adv_l.update(adv_loss.item(), inputs.size(0))
                    loss_reg_l.update(reg_loss.item(), inputs.size(0))
                    # top5.update(prec5[0], inputs.size(0))

                    if phase == "train":
                        # make the accumulated gradient to be the same scale as without the accumulation
                        # loss = loss / grad_steps


                        if mixed_prec:
                            with amp.scale_loss(
                                loss, optimizer
                            ) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                            
                        # loss.backward()

                        optimizer.step()
                        optimizer.zero_grad()
                        # if step % grad_steps == 0:
                        #     optimizer.step()
                        #     optimizer.zero_grad()

                        if step % 10 == 0:
                            print(f"{phase} took {batch_time.sum:.2f} sec ", end="| ")
                            print(f"loss = {losses.avg:.4f} ", end="| ")
                            print(f"adv loss = {loss_adv_l.avg:.4f} ", end="| ")
                            print(f"reg loss = {loss_reg_l.avg:.4f} ", end="| ")
                            print(f"fooling_ratio = {miss_rate/total_val_vid:.4f} ", end="| ")
                            perturbation =pert_model.get_perturbation()[0].data.cpu().numpy()
                            inf_norm= perturbation.__abs__().max()
                            thickness= perturbation.__abs__().mean()
                            roughness = (np.roll(perturbation, 1,1)-perturbation).__abs__().mean()
                            print(f"pert_thickness = {thickness:.4f} ", end="| ")
                            print(f"pert_roughness = {roughness:.4f} ", end="| ")
                            print(f"pert_inf_norm = {inf_norm:.4f} ", end=" ")
                            # if topk >= 5:
                            #     print(f"| top5_acc = {top5.avg:.4f}", end="")
                            print()
                    # measure elapsed time
                    batch_time.update(time() - end)
                    end = time()

            print(f"{phase} took {batch_time.sum:.2f} sec ", end="| ")
            print(f"loss = {losses.avg:.4f} ", end="| ")
            print(f"adv loss = {loss_adv_l.avg:.4f} ", end="| ")
            print(f"reg loss = {loss_reg_l.avg:.4f} ", end="| ")
            print(f"fooling_ratio = {miss_rate/total_val_vid:.4f} ", end="| ")
            perturbation =pert_model.get_perturbation()[0].data.cpu().numpy()
            inf_norm= perturbation.__abs__().max()
            thickness= perturbation.__abs__().mean()
            roughness = (np.roll(perturbation, 1,1)-perturbation).__abs__().mean()
            
            print(f"pert_thickness = {thickness:.4f} ", end="| ")
            print(f"pert_roughness = {roughness:.4f} ", end="| ")
            print(f"pert_inf_norm = {inf_norm:.4f} ", end=" ")

            # if topk >= 5:
            #     print(f"| top5_acc = {top5.avg:.4f}", end="")
            print()

            result[f"{phase}/time"] = batch_time.sum
            result[f"{phase}/loss"] = losses.avg
            result[f"{phase}/fooling_ratio"] = miss_rate/total_val_vid
            result[f"{phase}/pert_thickness"] = thickness
            result[f"{phase}/pert_roughness"] = roughness
            result[f"{phase}/inf_norm"] = inf_norm
            result[f"{phase}/perturbation"] = perturbation

        return result

    
    def fit_many_videos(
    self,
    lr: float,
    epochs: int,
    model_dir: str = "checkpoints",
    model_name: str = None,
    momentum: float = 0.95,
    weight_decay: float = 0.0001,
    mixed_prec: bool = False,
    use_one_cycle_policy: bool = False,
    warmup_pct: float = 0.3,
    lr_gamma: float = 0.1,
    lr_step_size: float = None,
    grad_steps: int = 2,
    save_model: bool = False,
    loss_params_dict= None,
    devices_ids =None
) -> None:
        """ The primary fit function """
        # set epochs
        self.epochs = epochs

        #set losses
        self.loss_handler =Losses(beta_1=loss_params_dict['beta_1'],
                                  lambda_=loss_params_dict['lambda_'],
                                  targeted=loss_params_dict['targeted_attack'],
                                  target_class=loss_params_dict['target_class_id'],
                                  improve_loss=loss_params_dict['improve_loss'],
                                  logits=loss_params_dict['use_logits'],
                                  attack_type=self.attack_type)

        self.metric_handler = Adversarial_metrics(targeted=loss_params_dict['targeted_attack'],
                                                  target_class=loss_params_dict['target_class_id'])

        # set model name
        if model_name is None:
            model_name = self.model_name

        os.makedirs(model_dir, exist_ok=True)

        data_loaders = {}
        dl = self.dataset.train_dl
        

        # Move model to gpu before constructing optimizers and amp.initialize
        device = torch_device()
        self.model.to(device)
        if devices_ids:
            count_devices = len(devices_ids)
        else:
            count_devices = num_devices()
            devices_ids = list(range(count_devices))
        
        torch.backends.cudnn.benchmark = True

        named_params_to_update = {}
        total_params = 0
        for name, param in self.pert_model.named_parameters():
            total_params += 1
            if param.requires_grad:
                named_params_to_update[name] = param

        print("Params to learn:")
        if len(named_params_to_update) == total_params:
            print("\tfull network")
        else:
            for name in named_params_to_update:
                print(f"\t{name}")

        # create optimizer
        # optimizer = optim.SGD(
        #     list(named_params_to_update.values()),
        #     lr=lr,
        #     momentum=momentum,
        #     weight_decay=weight_decay,
        # )

        optimizer = torch.optim.Adam(list(named_params_to_update.values()), lr=lr)
        # optimizer = torch.optim.SGD(list(named_params_to_update.values()), lr=lr, momentum=0.9)

        # Use mixed-precision if available
        # Currently, only O1 works with DataParallel: See issues https://github.com/NVIDIA/apex/issues/227
        if AMP_AVAILABLE and mixed_prec:
            # break if not AMP_AVAILABLE
            assert AMP_AVAILABLE
            # 'O0': Full FP32, 'O1': Conservative, 'O2': Standard, 'O3': Full FP16
            self.model, optimizer = amp.initialize(
                self.model,
                optimizer,
                opt_level="O1",
                loss_scale="dynamic",
                # keep_batchnorm_fp32=True doesn't work on 'O1'
            )

        # Learning rate scheduler
        if use_one_cycle_policy:
            # Use warmup with the one-cycle policy
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=self.epochs,
                pct_start=warmup_pct,
                base_momentum=0.9 * momentum,
                max_momentum=momentum,
            )
        else:
            # Simple step-decay
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=lr_step_size, gamma=lr_gamma,
            )

        # DataParallel after amp.initialize
        model = (
            nn.DataParallel(self.model,device_ids=devices_ids) if count_devices > 1 else self.model
        )

        # criterion = nn.CrossEntropyLoss().to(device)

        # set num classes
        topk = 5
        if topk >= self.num_classes:
            topk = self.num_classes
        
        
        
        targeted_attack = loss_params_dict['targeted_attack']
        target_class_id = loss_params_dict['target_class_id']
        for vid_num, (inputs, target,  vid_path) in enumerate(dl, start=0):
            # optimizer.load_state_dict(optimizer.state_dict())
            
            print(
                f"Video_attack {vid_num} ========================================================="
            )

            class_name = self.label_id_to_text[target]
            dest_path = os.path.join(
                        model_dir,
                        '{vid_name}_@{claas_name}.npy'.format(vid_name=vid_path[0].split('/')[-1],
                                                                         claas_name=class_name.replace(' ','_')))
            if os.path.exists(dest_path):
                res_dict = np.load(dest_path, allow_pickle=True)

                res_dict = res_dict.tolist()
                if res_dict is None:
                    continue
                is_adversarial = np.array(res_dict['is_adversarial'])

                if is_adversarial.any() == True:
                    print('video {} exist, skipping...'.format(vid_path[0].split('/')[-1]))
                    continue
            else:
                if save_model:
                    np.save(dest_path, None)


            optimizer.param_groups[0]['params'][0].data =torch.rand(size=self.pert_model.size,
                                                                    requires_grad=self.pert_model.requires_grad,
                                                                    device=self.pert_model.device).mul(2).sub(1).mul(0.005)
            self.pert_model.dynamic_max_norm=self.pert_model.max_norm
            result_dict =self.fit_single_video_attack(
                    model = model,
                    pert_model= self.pert_model,
                    inputs = inputs,
                    target = target,
                    device =device,
                    criterion =self.loss_handler,
                    metric=self.metric_handler,
                    optimizer=optimizer,
                    grad_steps=grad_steps,
                    mixed_prec=mixed_prec,
                    topk=topk,
                    n_iter=3000,
                    targeted_attack= targeted_attack,
                    target_class_id=target_class_id)
            
            if result_dict ==None:
                continue
            
            

            if save_model:
                # self.save(
                #     os.path.join(
                #         model_dir,
                #         "{model_name}_{vid_num}.pt".format(
                #             model_name=model_name, epoch=str(e).zfill(5),
                #         ),
                #     )
                # )
                np.save(dest_path ,result_dict)
        # self.plot_precision_loss_curves()
    
    @staticmethod
    def fit_single_video_attack(
        model,
        pert_model,
        inputs,
        target,
        device,
        criterion,
        metric,
        optimizer,
        grad_steps: int = 1,
        mixed_prec: bool = False,
        topk: int = 5,
        n_iter: int=3000,
        targeted_attack: bool= False,
        target_class_id: int=None
    ) -> Dict[str, Any]:

        target_class_id = target_class_id
        to_plot = True
        if mixed_prec and not AMP_AVAILABLE:
            warnings.warn(
                """
                NVIDIA apex module is not installed. Cannot use
                mixed-precision. Turning off mixed-precision.
                """
            )
            mixed_prec = False

        result = OrderedDict()

        # collect metrics
        batch_time = AverageMeter()
        losses = AverageMeter()
        loss_adv_l = AverageMeter()
        loss_reg_l = AverageMeter()
        fooling_ratio_l = AverageMeter()
        top5 = AverageMeter()

        end = time()
        step =0
        
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        outputs_no_adv = model([inputs,False])
        scores_no_adv = nn.functional.softmax(outputs_no_adv, dim=1)
        
        if not scores_no_adv.argmax().eq(target):

            return None

        is_adversarial = False

        max_prob_l=[]
        thickness_l=[]
        roughness_l=[]
        correct_cls_prob_l=[]
        is_adversarial_l=[]
        perturbation_l=[]

        new_chance=0
        if to_plot:
            fig, (ax1, ax3) = plt.subplots(2,1)
            ax1.set_xlabel('iter (#)')
            ax1.set_ylabel('probability', color='g')
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel('MAP[%]', color='b')
            
            # fig2, ax3 = plt.subplots()
            ax3.set_xlabel('iter (#)')
            ax3.set_ylabel('loss', color='g')
        
        while(step < n_iter or not is_adversarial):
            
            if step % 10 == 0:
                print(f"batch {step} of {n_iter}")

            if step >3000:
                new_chance+=1
                pert_model.dynamic_max_norm*=1.3
                step =0
            if new_chance==4:
                break
            # with torch.no_grad():
            #     outputs_no_adv = model([inputs,False])
            #     scores_no_adv = nn.functional.softmax(outputs_no_adv, dim=1)
                
            with torch.set_grad_enabled(True):
                # compute output
                outputs = model([inputs, True])

                scores = nn.functional.softmax(outputs, dim=1)
                outputs = outputs.type(scores.type())

                perturbation,_ = pert_model.get_perturbation()
                loss, adv_loss, reg_loss = criterion(target,outputs,scores,perturbation)
                
                adv_class= scores.argmax()                
            
               

                losses.update(loss.item(), inputs.size(0))
                loss_adv_l.update(adv_loss.item(), inputs.size(0))
                loss_reg_l.update(reg_loss.item(), inputs.size(0))
                # top5.update(prec5[0], inputs.size(0))

                if mixed_prec:
                    with amp.scale_loss(
                        loss, optimizer
                    ) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                    
                # loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                # if step % grad_steps == 0:
                #     optimizer.step()
                #     optimizer.zero_grad()
                
                is_adversarial= adv_class.cpu().data==target_class_id if targeted_attack else not adv_class.equal(target[0])
                is_adversarial_l.append(is_adversarial)


                perturbation,_ = pert_model.get_perturbation()
                perturbation = perturbation.data.cpu().numpy()
                perturbation_l.append(perturbation)

                inf_norm= perturbation.__abs__().max()
                thickness = perturbation.__abs__().mean()
                roughness = (np.roll(perturbation, 1,1)-perturbation).__abs__().mean()
                thickness_l.append(thickness)
                roughness_l.append(roughness)
                
                max_prob_l.append(scores.max().data.cpu().numpy().item())
                correct_cls_prob_l.append(criterion.label_prob.data.cpu().numpy()[0].item())

                if step % 100 == 0:
                    
                    if to_plot:
                        
                        lns2 = ax1.plot(correct_cls_prob_l, 'r', label='original class')
                        lns1 = ax1.semilogx(max_prob_l, '--g', label='max class')
                        ax1.tick_params(axis='y', labelcolor='g')


                        lns3 = ax2.plot(roughness_l, 'b', label='roughness')
                        lns4 = ax2.plot(thickness_l, '--b', label='thickness')
                        ax2.tick_params(axis='y', labelcolor='b')
                        
                        lns = lns1 + lns2 + lns3 + lns4
                        labs = [l.get_label() for l in lns]
                        ax1.legend(lns, labs, loc=6)
                        ax2.grid(True)
                        
                        lns5= ax3.semilogx(losses.vals, 'b', label='total_loss')
                        lns6= ax3.plot(loss_adv_l.vals, 'r', label='adv_loss')
                        lns7= ax3.plot(loss_reg_l.vals, 'g', label='reg_loss')
                        ax3.set_yscale('log')
                        lns = lns5 + lns6 + lns7
                        labs = [l.get_label() for l in lns]
                        ax3.legend(lns, labs, loc=6)
                        ax3.grid(True)
                        
                        fig.tight_layout()  # otherwise the right y-label is slightly clipped
                        plt.show(block=False)
    

                        plt.pause(0.1)
                # if 0:
                    print(f" took {batch_time.sum:.2f} sec ", end="| ")
                    print(f"loss = {losses.val:.4f} ", end="| ")
                    print(f"adv loss = {loss_adv_l.val:.4f} ", end="| ")
                    print(f"reg loss = {loss_reg_l.val:.4f} ", end="| ")
                    # print(f"fooling_ratio = {fooling_ratio_l.val:.4f} ", end="| ")

                    print(f"pert_thickness = {thickness:.4f} ", end="| ")
                    print(f"pert_roughness = {roughness:.4f} ", end="| ")
                    print(f"pert_inf_norm = {inf_norm:.4f} ", end=" ")
                    # if topk >= 5:
                    #     print(f"| top5_acc = {top5.avg:.4f}", end="")
                    print()
                # measure elapsed time
                batch_time.update(time() - end)
                end = time()

                step+=1

        print(f"took {batch_time.sum:.2f} sec ", end="| ")
        print(f"loss = {losses.val:.4f} ", end="| ")
        print(f"adv loss = {loss_adv_l.val:.4f} ", end="| ")
        print(f"reg loss = {loss_reg_l.val:.4f} ", end="| ")
        # print(f"fooling_ratio = {fooling_ratio_l.val:.4f} ", end="| ")
        perturbation, _ = pert_model.get_perturbation()
        perturbation = perturbation.data.cpu().numpy()
        inf_norm= perturbation.__abs__().max()
        thickness= perturbation.__abs__().mean()
        roughness = (np.roll(perturbation, 1,1)-perturbation).__abs__().mean()
        
        print(f"pert_thickness = {thickness:.4f} ", end="| ")
        print(f"pert_roughness = {roughness:.4f} ", end="| ")
        print(f"pert_inf_norm = {inf_norm:.4f} ", end=" ")

        # if topk >= 5:
        #     print(f"| top5_acc = {top5.avg:.4f}", end="")
        print()

        # result[f"time"] = batch_time.sum
        result[f"loss/total"] = losses.vals
        result[f"loss/adv_loss"] = loss_adv_l.vals
        result[f"loss/reg_loss"] = loss_reg_l.vals
        result[f"perturbation/thickness"] = thickness_l
        result[f"perturbation/roughness"] = roughness_l
        result[f"perturbation/inf_norm"] = inf_norm
        result[f"perturbation"] = perturbation_l
        result[f"prob_clean_input"] = outputs_no_adv
        result[f"label"] = target.data.cpu().numpy()
        result[f"is_adversarial"] = is_adversarial_l

        return result        
    def plot_precision_loss_curves(
        self, figsize: Tuple[int, int] = (10, 5)
    ) -> None:
        """ Plot training loss and accuracy from calling `fit` on the test set. """
        assert len(self.results) > 0

        fig = plt.figure(figsize=figsize)
        valid_losses = [dic["valid/loss"] for dic in self.results]
        valid_top1 = [float(dic["valid/top1"]) for dic in self.results]

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_xlim([0, self.epochs - 1])
        ax1.set_xticks(range(0, self.epochs))
        ax1.set_xlabel("epochs")
        ax1.set_ylabel("loss", color="g")
        ax1.plot(valid_losses, "g-")
        ax2 = ax1.twinx()
        ax2.set_ylabel("top1 %acc", color="b")
        ax2.plot(valid_top1, "b-")
        fig.suptitle("Loss and Average Precision (AP) over Epochs")

    def evaluate(
        self,
        num_samples: int = 10,
        report_every: int = 100,
        train_or_test: str = "test",
    ) -> None:
        """ eval code for validation/test set and saves the evaluation results in self.results.

        Args:
            num_samples: number of samples (clips) of the validation set to test
            report_every: print line of results every n times
            train_or_test: use train or test set
        """
        # asset train or test valid
        assert train_or_test in ["train", "test"]

        # set device and num_gpus
        num_gpus = num_devices()
        device = torch_device()
        torch.backends.cudnn.benchmark = True if cuda.is_available() else False

        # init model with gpu (or not)
        self.model.to(device)
        if num_gpus > 1:
            self.model = nn.DataParallel(model)
        self.model.eval()

        # set train or test
        ds = (
            self.dataset.test_ds
            if train_or_test == "test"
            else self.dataset.train_ds
        )

        # set num_samples
        ds.dataset.num_samples = num_samples
        print(
            f"{len(self.dataset.test_ds)} samples of {self.dataset.test_ds[0][0][0].shape}"
        )

        # Loop over all examples in the test set and compute accuracies
        ret = dict(
            infer_times=[],
            video_preds=[],
            video_trues=[],
            clip_preds=[],
            clip_trues=[],
        )
        report_every = 100

        # inference
        with torch.no_grad():
            for i in range(
                1, len(ds)
            ):  # [::10]:  # Skip some examples to speed up accuracy computation
                if i % report_every == 0:
                    print(
                        f"Processsing {i} of {len(self.dataset.test_ds)} samples.."
                    )

                # Get model inputs
                inputs, label = ds[i]
                inputs = inputs.to(device, non_blocking=True)

                # Run inference
                start_time = time()
                outputs = self.model(inputs)
                outputs = outputs.cpu().numpy()
                infer_time = time() - start_time
                ret["infer_times"].append(infer_time)

                # Store results
                ret["video_preds"].append(outputs.sum(axis=0).argmax())
                ret["video_trues"].append(label)
                ret["clip_preds"].extend(outputs.argmax(axis=1))
                ret["clip_trues"].extend([label] * num_samples)

        print(
            f"Avg. inference time per video ({len(ds)} clips) =",
            round(np.array(ret["infer_times"]).mean() * 1000, 2),
            "ms",
        )
        print(
            "Video prediction accuracy =",
            round(accuracy_score(ret["video_trues"], ret["video_preds"]), 2),
        )
        print(
            "Clip prediction accuracy =",
            round(accuracy_score(ret["clip_trues"], ret["clip_preds"]), 2),
        )
        return ret

    def _predict(self, frames, transform):
        """Runs prediction on frames applying transforms before predictions."""
        clip = torch.from_numpy(np.array(frames))
        # Transform frames and append batch dim
        sample = torch.unsqueeze(transform(clip), 0)
        sample = sample.to(torch_device())
        output = self.model(sample)
        scores = nn.functional.softmax(output, dim=1).data.cpu().numpy()[0]
        return scores

    def _filter_labels(
        self,
        id_score_dict: dict,
        labels: List[str],
        threshold: float = 0.0,
        target_labels: List[str] = None,
        filter_labels: List[str] = None,
    ) -> Dict[str, int]:
        """ Given the predictions, filter out the noise based on threshold,
        target labels and filter labels.

        Arg:
            id_score_dict: dictionary of predictions
            labels: all labels
            threshold: the min threshold to keep prediction
            target_labels: exclude any labels not in target labels
            filter_labels: exclude any labels in filter labels

        Returns
            A dictionary of labels and scores
        """
        # Show only interested actions (target_labels) with a confidence score >= threshold
        result = {}
        for i, s in id_score_dict.items():
            label = labels[i]
            if (
                (s < threshold)
                or (target_labels is not None and label not in target_labels)
                or (filter_labels is not None and label in filter_labels)
            ):
                continue

            if label in result:
                result[label] += s
            else:
                result[label] = s

        return result

    def predict_frames(
        self,
        window: deque,
        scores_cache: deque,
        scores_sum: np.ndarray,
        is_ready: list,
        averaging_size: int,
        score_threshold: float,
        labels: List[str],
        target_labels: List[str],
        transforms: Compose,
        update_println: Callable,
    ) -> None:
        """ Predicts frames """
        # set model device and to eval mode
        self.model.to(torch_device())
        self.model.eval()

        # score
        t = time()
        scores = self._predict(window, transforms)
        print("max: {}, {}".format(scores.max(),scores.argmax()))
        dur = time() - t

        # Averaging scores across clips (dense prediction)
        scores_cache.append(scores)
        scores_sum += scores

        if len(scores_cache) == averaging_size:
            scores_avg = scores_sum / averaging_size

            if len(labels) >= 5:
                num_labels = 5
            else:
                num_labels = len(labels) - 1

            top5_id_score_dict = {
                i: scores_avg[i]
                for i in (-scores_avg).argpartition(num_labels - 1)[
                    :num_labels
                ]
            }
            top5_label_score_dict = self._filter_labels(
                top5_id_score_dict,
                labels,
                threshold=score_threshold,
                target_labels=target_labels,
            )
            top5 = sorted(top5_label_score_dict.items(), key=lambda kv: -kv[1])

            # fps and preds
            println = (
                f"{1 // dur} fps"
                + "<p style='font-size:20px'>"
                + "<br>".join([f"{k} ({v:.3f})" for k, v in top5])
                + "</p>"
            )

            # Plot final results nicely
            update_println(println)
            scores_sum -= scores_cache.popleft()

        # Inference done. Ready to run on the next frames.
        window.popleft()
        if is_ready:
            is_ready[0] = True

    def predict_video(
        self,
        video_fpath: str,
        labels: List[str] = None,
        averaging_size: int = 5,
        score_threshold: float = 0.025,
        target_labels: List[str] = None,
        transforms: Compose = None,
    ) -> None:
        """Load video and show frames and inference results while displaying the results
        """
        # set up video reader
        video_reader = decord.VideoReader(video_fpath)
        print(f"Total frames = {len(video_reader)}")

        # set up ipython jupyter display
        d_video = IPython.display.display("", display_id=1)
        d_caption = IPython.display.display("Preparing...", display_id=2)

        # set vars
        is_ready = [True]
        window = deque()
        scores_cache = deque()

        # use labels if given, else see if we have labels from our dataset
        if not labels:
            if self.dataset.classes:
                labels = self.dataset.classes
            else:
                raise ("No labels found, add labels argument.")
        scores_sum = np.zeros(len(labels))

        # set up transforms
        if not transforms:
            transforms = get_transforms(train=False)

        # set up print function
        def update_println(println):
            d_caption.update(IPython.display.HTML(println))

        while True:
            try:
                frame = video_reader.next().asnumpy()
                if len(frame.shape) != 3:
                    break

                # Start an inference thread when ready
                if is_ready[0]:
                    window.append(frame)
                    if len(window) == self.sample_length:
                        is_ready[0] = False
                        Thread(
                            target=self.predict_frames,
                            args=(
                                window,
                                scores_cache,
                                scores_sum,
                                is_ready,
                                averaging_size,
                                score_threshold,
                                labels,
                                target_labels,
                                transforms,
                                update_println,
                            ),
                        ).start()

                # Show video preview
                f = io.BytesIO()
                im = Image.fromarray(frame)
                im.save(f, "jpeg")

                # resize frames to avoid flicker for windows
                w, h = frame.shape[0], frame.shape[1]
                scale = 300.0 / max(w, h)
                w = round(w * scale)
                h = round(h * scale)
                im = im.resize((h, w))

                d_video.update(IPython.display.Image(data=f.getvalue()))
                sleep(0.03)
            except Exception:
                break

    def save(self, model_path: Union[Path, str]) -> None:
        """ Save the model to a path on disk. """
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_name: str, model_dir: str = "checkpoints") -> None:
        """
        TODO accept epoch. If None, load the latest model.
        :param model_name: Model name format should be 'name_0EE' where E is the epoch
        :param model_dir: By default, 'checkpoints'
        :return:
        """
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, f"{model_name}.pt"))
        )



class VideoLearner(object):
    """ Video recognition learner object that handles training loop and evaluation. """

    def __init__(
        self,
        dataset: VideoDataset = None,
        num_classes: int = None,  # ie 51 for hmdb51
        base_model: str = "ig65m",  # or "kinetics"
        sample_length: int = None,
    ) -> None:
        """ By default, the Video Learner will use a R2plus1D model. Pass in
        a dataset of type Video Dataset and the Video Learner will intialize
        the model.

        Args:
            dataset: the datset to use for this model
            num_class: the number of actions/classifications
            base_model: the R2plus1D model is based on either ig65m or
            kinetics. By default it will use the weights from ig65m since it
            tends attain higher results.
        """
        # set empty - populated when fit is called
        self.results = []

        # set num classes
        self.num_classes = num_classes

        if dataset:
            self.dataset = dataset
            self.sample_length = self.dataset.sample_length
        else:
            assert sample_length == 8 or sample_length == 32
            self.sample_length = sample_length

        self.model, self.model_name = self.init_model(
            self.sample_length, base_model, num_classes,
        )

    @staticmethod
    def init_model(
        sample_length: int, base_model: str, num_classes: int = None
    ) -> torchvision.models.video.resnet.VideoResNet:
        """
        Initializes the model by loading it using torch's `hub.load`
        functionality. Uses the model from TORCH_R2PLUS1D.

        Args:
            sample_length: Number of consecutive frames to sample from a video (i.e. clip length).
            base_model: the R2plus1D model is based on either ig65m or kinetics.
            num_classes: the number of classes/actions

        Returns:
            Load a model from a github repo, with pretrained weights
        """
        if base_model not in ("ig65m", "kinetics"):
            raise ValueError(
                f"Not supported model {base_model}. Should be 'ig65m' or 'kinetics'"
            )

        # Decide if to use pre-trained weights for DNN trained using 8 or for 32 frames
        model_name = f"r2plus1d_34_{sample_length}_{base_model}"

        print(f"Loading {model_name} model")

        model = torch.hub.load(
            TORCH_R2PLUS1D,
            model_name,
            num_classes=MODELS[model_name],
            pretrained=True,
        )

        # Replace head
        if num_classes is not None:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model, model_name

    def freeze(self) -> None:
        """Freeze model except the last layer"""
        self._set_requires_grad(False)
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze(self) -> None:
        """Unfreeze all layers in model"""
        self._set_requires_grad(True)

    def _set_requires_grad(self, requires_grad=True) -> None:
        """ sets requires grad """
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def fit(
        self,
        lr: float,
        epochs: int,
        model_dir: str = "checkpoints",
        model_name: str = None,
        momentum: float = 0.95,
        weight_decay: float = 0.0001,
        mixed_prec: bool = False,
        use_one_cycle_policy: bool = False,
        warmup_pct: float = 0.3,
        lr_gamma: float = 0.1,
        lr_step_size: float = None,
        grad_steps: int = 2,
        save_model: bool = False,
    ) -> None:
        """ The primary fit function """
        # set epochs
        self.epochs = epochs

        # set lr_step_size based on epochs
        if lr_step_size is None:
            lr_step_size = np.ceil(2 / 3 * self.epochs)

        # set model name
        if model_name is None:
            model_name = self.model_name

        os.makedirs(model_dir, exist_ok=True)

        data_loaders = {}
        data_loaders["train"] = self.dataset.train_dl
        data_loaders["valid"] = self.dataset.test_dl

        # Move model to gpu before constructing optimizers and amp.initialize
        device = torch_device()
        self.model.to(device)
        count_devices = num_devices()
        torch.backends.cudnn.benchmark = True

        named_params_to_update = {}
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += 1
            if param.requires_grad:
                named_params_to_update[name] = param

        print("Params to learn:")
        if len(named_params_to_update) == total_params:
            print("\tfull network")
        else:
            for name in named_params_to_update:
                print(f"\t{name}")

        # create optimizer
        optimizer = optim.SGD(
            list(named_params_to_update.values()),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        # Use mixed-precision if available
        # Currently, only O1 works with DataParallel: See issues https://github.com/NVIDIA/apex/issues/227
        if AMP_AVAILABLE and mixed_prec:
            # break if not AMP_AVAILABLE
            assert AMP_AVAILABLE
            # 'O0': Full FP32, 'O1': Conservative, 'O2': Standard, 'O3': Full FP16
            self.model, optimizer = amp.initialize(
                self.model,
                optimizer,
                opt_level="O1",
                loss_scale="dynamic",
                # keep_batchnorm_fp32=True doesn't work on 'O1'
            )

        # Learning rate scheduler
        if use_one_cycle_policy:
            # Use warmup with the one-cycle policy
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=self.epochs,
                pct_start=warmup_pct,
                base_momentum=0.9 * momentum,
                max_momentum=momentum,
            )
        else:
            # Simple step-decay
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=lr_step_size, gamma=lr_gamma,
            )

        # DataParallel after amp.initialize
        model = (
            nn.DataParallel(self.model) if count_devices > 1 else self.model
        )

        criterion = nn.CrossEntropyLoss().to(device)

        # set num classes
        topk = 5
        if topk >= self.num_classes:
            topk = self.num_classes

        for e in range(1, self.epochs + 1):
            print(
                f"Epoch {e} ========================================================="
            )
            print(f"lr={scheduler.get_lr()}")

            self.results.append(
                self.train_an_epoch(
                    model,
                    data_loaders,
                    device,
                    criterion,
                    optimizer,
                    grad_steps=grad_steps,
                    mixed_prec=mixed_prec,
                    topk=topk,
                )
            )

            scheduler.step()

            if save_model:
                self.save(
                    os.path.join(
                        model_dir,
                        "{model_name}_{self.epoch}.pt".format(
                            model_name=model_name, epoch=str(e).zfill(3),
                        ),
                    )
                )
        self.plot_precision_loss_curves()

    @staticmethod
    def train_an_epoch(
        model,
        data_loaders,
        device,
        criterion,
        optimizer,
        grad_steps: int = 1,
        mixed_prec: bool = False,
        topk: int = 5,
    ) -> Dict[str, Any]:
        """Train / validate a model for one epoch.

        Args:
            model: the model to use to train
            data_loaders: dict {'train': train_dl, 'valid': valid_dl}
            device: gpu or not
            criterion: TODO
            optimizer: TODO
            grad_steps: If > 1, use gradient accumulation. Useful for larger batching
            mixed_prec: If True, use FP16 + FP32 mixed precision via NVIDIA apex.amp
            topk: top k classes

        Return:
            dict {
                'train/time': batch_time.avg,
                'train/loss': losses.avg,
                'train/top1': top1.avg,
                'train/top5': top5.avg,
                'valid/time': ...
            }
        """
        if mixed_prec and not AMP_AVAILABLE:
            warnings.warn(
                """
                NVIDIA apex module is not installed. Cannot use
                mixed-precision. Turning off mixed-precision.
                """
            )
            mixed_prec = False

        result = OrderedDict()
        for phase in ["train", "valid"]:
            # switch mode
            if phase == "train":
                model.train()
            else:
                model.eval()

            # set loader
            dl = data_loaders[phase]

            # collect metrics
            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            end = time()
            for step, (inputs, target) in enumerate(dl, start=1):
                if step % 10 == 0:
                    print(f" Phase {phase}: batch {step} of {len(dl)}")
                inputs = inputs.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                with torch.set_grad_enabled(phase == "train"):
                    # compute output
                    outputs = model(inputs)
                    loss = criterion(outputs, target)

                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(outputs, target, topk=(1, topk))

                    losses.update(loss.item(), inputs.size(0))
                    top1.update(prec1[0], inputs.size(0))
                    top5.update(prec5[0], inputs.size(0))

                    if phase == "train":
                        # make the accumulated gradient to be the same scale as without the accumulation
                        loss = loss / grad_steps

                        if mixed_prec:
                            with amp.scale_loss(
                                loss, optimizer
                            ) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        if step % grad_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                    # measure elapsed time
                    batch_time.update(time() - end)
                    end = time()

            print(f"{phase} took {batch_time.sum:.2f} sec ", end="| ")
            print(f"loss = {losses.avg:.4f} ", end="| ")
            print(f"fooling_ratio = {top1.avg:.4f} ", end=" ")
            # if topk >= 5:
            #     print(f"| top5_acc = {top5.avg:.4f}", end="")
            # print()

            result[f"{phase}/time"] = batch_time.sum
            result[f"{phase}/loss"] = losses.avg
            result[f"{phase}/top1"] = top1.avg
            result[f"{phase}/top5"] = top5.avg

        return result

    def plot_precision_loss_curves(
        self, figsize: Tuple[int, int] = (10, 5)
    ) -> None:
        """ Plot training loss and accuracy from calling `fit` on the test set. """
        assert len(self.results) > 0

        fig = plt.figure(figsize=figsize)
        valid_losses = [dic["valid/loss"] for dic in self.results]
        valid_top1 = [float(dic["valid/top1"]) for dic in self.results]

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_xlim([0, self.epochs - 1])
        ax1.set_xticks(range(0, self.epochs))
        ax1.set_xlabel("epochs")
        ax1.set_ylabel("loss", color="g")
        ax1.plot(valid_losses, "g-")
        ax2 = ax1.twinx()
        ax2.set_ylabel("top1 %acc", color="b")
        ax2.plot(valid_top1, "b-")
        fig.suptitle("Loss and Average Precision (AP) over Epochs")

    def evaluate(
        self,
        num_samples: int = 10,
        report_every: int = 100,
        train_or_test: str = "test",
    ) -> None:
        """ eval code for validation/test set and saves the evaluation results in self.results.

        Args:
            num_samples: number of samples (clips) of the validation set to test
            report_every: print line of results every n times
            train_or_test: use train or test set
        """
        # asset train or test valid
        assert train_or_test in ["train", "test"]

        # set device and num_gpus
        num_gpus = num_devices()
        device = torch_device()
        torch.backends.cudnn.benchmark = True if cuda.is_available() else False

        # init model with gpu (or not)
        self.model.to(device)
        if num_gpus > 1:
            self.model = nn.DataParallel(model)
        self.model.eval()

        # set train or test
        ds = (
            self.dataset.test_ds
            if train_or_test == "test"
            else self.dataset.train_ds
        )

        # set num_samples
        ds.dataset.num_samples = num_samples
        print(
            f"{len(self.dataset.test_ds)} samples of {self.dataset.test_ds[0][0][0].shape}"
        )

        # Loop over all examples in the test set and compute accuracies
        ret = dict(
            infer_times=[],
            video_preds=[],
            video_trues=[],
            clip_preds=[],
            clip_trues=[],
        )
        report_every = 100

        # inference
        with torch.no_grad():
            for i in range(
                1, len(ds)
            ):  # [::10]:  # Skip some examples to speed up accuracy computation
                if i % report_every == 0:
                    print(
                        f"Processsing {i} of {len(self.dataset.test_ds)} samples.."
                    )

                # Get model inputs
                inputs, label = ds[i]
                inputs = inputs.to(device, non_blocking=True)

                # Run inference
                start_time = time()
                outputs = self.model(inputs)
                outputs = outputs.cpu().numpy()
                infer_time = time() - start_time
                ret["infer_times"].append(infer_time)

                # Store results
                ret["video_preds"].append(outputs.sum(axis=0).argmax())
                ret["video_trues"].append(label)
                ret["clip_preds"].extend(outputs.argmax(axis=1))
                ret["clip_trues"].extend([label] * num_samples)

        print(
            f"Avg. inference time per video ({len(ds)} clips) =",
            round(np.array(ret["infer_times"]).mean() * 1000, 2),
            "ms",
        )
        print(
            "Video prediction accuracy =",
            round(accuracy_score(ret["video_trues"], ret["video_preds"]), 2),
        )
        print(
            "Clip prediction accuracy =",
            round(accuracy_score(ret["clip_trues"], ret["clip_preds"]), 2),
        )
        return ret

    def _predict(self, frames, transform):
        """Runs prediction on frames applying transforms before predictions."""
        clip = torch.from_numpy(np.array(frames))
        # Transform frames and append batch dim
        sample = torch.unsqueeze(transform(clip), 0)
        sample = sample.to(torch_device())
        output = self.model(sample)
        scores = nn.functional.softmax(output, dim=1).data.cpu().numpy()[0]
        return scores

    def _filter_labels(
        self,
        id_score_dict: dict,
        labels: List[str],
        threshold: float = 0.0,
        target_labels: List[str] = None,
        filter_labels: List[str] = None,
    ) -> Dict[str, int]:
        """ Given the predictions, filter out the noise based on threshold,
        target labels and filter labels.

        Arg:
            id_score_dict: dictionary of predictions
            labels: all labels
            threshold: the min threshold to keep prediction
            target_labels: exclude any labels not in target labels
            filter_labels: exclude any labels in filter labels

        Returns
            A dictionary of labels and scores
        """
        # Show only interested actions (target_labels) with a confidence score >= threshold
        result = {}
        for i, s in id_score_dict.items():
            label = labels[i]
            if (
                (s < threshold)
                or (target_labels is not None and label not in target_labels)
                or (filter_labels is not None and label in filter_labels)
            ):
                continue

            if label in result:
                result[label] += s
            else:
                result[label] = s

        return result

    def predict_frames(
        self,
        window: deque,
        scores_cache: deque,
        scores_sum: np.ndarray,
        is_ready: list,
        averaging_size: int,
        score_threshold: float,
        labels: List[str],
        target_labels: List[str],
        transforms: Compose,
        update_println: Callable,
    ) -> None:
        """ Predicts frames """
        # set model device and to eval mode
        self.model.to(torch_device())
        self.model.eval()

        # score
        t = time()
        scores = self._predict(window, transforms)
        print("max: {}, {}".format(scores.max(),scores.argmax()))
        dur = time() - t

        # Averaging scores across clips (dense prediction)
        scores_cache.append(scores)
        scores_sum += scores

        if len(scores_cache) == averaging_size:
            scores_avg = scores_sum / averaging_size

            if len(labels) >= 5:
                num_labels = 5
            else:
                num_labels = len(labels) - 1

            top5_id_score_dict = {
                i: scores_avg[i]
                for i in (-scores_avg).argpartition(num_labels - 1)[
                    :num_labels
                ]
            }
            top5_label_score_dict = self._filter_labels(
                top5_id_score_dict,
                labels,
                threshold=score_threshold,
                target_labels=target_labels,
            )
            top5 = sorted(top5_label_score_dict.items(), key=lambda kv: -kv[1])

            # fps and preds
            println = (
                f"{1 // dur} fps"
                + "<p style='font-size:20px'>"
                + "<br>".join([f"{k} ({v:.3f})" for k, v in top5])
                + "</p>"
            )

            # Plot final results nicely
            update_println(println)
            scores_sum -= scores_cache.popleft()

        # Inference done. Ready to run on the next frames.
        window.popleft()
        if is_ready:
            is_ready[0] = True

    def predict_video(
        self,
        video_fpath: str,
        labels: List[str] = None,
        averaging_size: int = 5,
        score_threshold: float = 0.025,
        target_labels: List[str] = None,
        transforms: Compose = None,
    ) -> None:
        """Load video and show frames and inference results while displaying the results
        """
        # set up video reader
        video_reader = decord.VideoReader(video_fpath)
        print(f"Total frames = {len(video_reader)}")

        # set up ipython jupyter display
        d_video = IPython.display.display("", display_id=1)
        d_caption = IPython.display.display("Preparing...", display_id=2)

        # set vars
        is_ready = [True]
        window = deque()
        scores_cache = deque()

        # use labels if given, else see if we have labels from our dataset
        if not labels:
            if self.dataset.classes:
                labels = self.dataset.classes
            else:
                raise ("No labels found, add labels argument.")
        scores_sum = np.zeros(len(labels))

        # set up transforms
        if not transforms:
            transforms = get_transforms(train=False)

        # set up print function
        def update_println(println):
            d_caption.update(IPython.display.HTML(println))

        while True:
            try:
                frame = video_reader.next().asnumpy()
                if len(frame.shape) != 3:
                    break

                # Start an inference thread when ready
                if is_ready[0]:
                    window.append(frame)
                    if len(window) == self.sample_length:
                        is_ready[0] = False
                        Thread(
                            target=self.predict_frames,
                            args=(
                                window,
                                scores_cache,
                                scores_sum,
                                is_ready,
                                averaging_size,
                                score_threshold,
                                labels,
                                target_labels,
                                transforms,
                                update_println,
                            ),
                        ).start()

                # Show video preview
                f = io.BytesIO()
                im = Image.fromarray(frame)
                im.save(f, "jpeg")

                # resize frames to avoid flicker for windows
                w, h = frame.shape[0], frame.shape[1]
                scale = 300.0 / max(w, h)
                w = round(w * scale)
                h = round(h * scale)
                im = im.resize((h, w))

                d_video.update(IPython.display.Image(data=f.getvalue()))
                sleep(0.03)
            except Exception:
                break

    def save(self, model_path: Union[Path, str]) -> None:
        """ Save the model to a path on disk. """
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_name: str, model_dir: str = "checkpoints") -> None:
        """
        TODO accept epoch. If None, load the latest model.
        :param model_name: Model name format should be 'name_0EE' where E is the epoch
        :param model_dir: By default, 'checkpoints'
        :return:
        """
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, f"{model_name}.pt"))
        )




##

