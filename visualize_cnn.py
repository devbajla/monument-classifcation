"""
EECS 445 - Introduction to Machine Learning
Fall 2023  - Project 2
Grad-CAM Visualization
    This script generates a heat map on top of the original image for 20 
    sample images in the dataset to help visualize what is learned by
    the convolutional networks.

    Output files will be titled CNN_viz1_<number>.png

    Usage: python visualize_cc.py

Original credit to:
Author:   Kazuto Nakashima
URL:      http://kazuto1011.github.io
Created:  2017-05-26
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from dataset import get_train_val_test_loaders
from model.target import Target
from model.source import Source
from train_common import *
from utils import config
import utils
from collections import OrderedDict
from torch.nn import functional as F
from imageio.v3 import imread


def save_gradcam(gcam, original_image, axarr, i):
    cmap = cm.viridis(np.squeeze(gcam.numpy()))[..., :3] * 255.0
    raw_image = (
        (
            (original_image - original_image.min())
            / (original_image.max() - original_image.min())
        )
        * 255
    ).astype("uint8")
    gcam = (cmap.astype(np.float64) + raw_image.astype(np.float64)) / 2
    axarr[1].imshow(np.uint8(gcam))
    axarr[1].axis("off")
    plt.savefig("CNN_viz1_{}.png".format(i), dpi=200, bbox_inches="tight")


class _BaseWrapper(object):
    """
    Please modify forward() and backward() according to your task.
    """

    def __init__(self, model, fmaps=None):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(image)
        if type(self.logits) is tuple:
            self.logits = self.logits[0]
        self.probs = torch.nn.Sigmoid()(self.logits)
        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        """
        Class-specific backpropagation
        Either way works:
        1. self.logits.backward(gradient=one_hot, retain_graph=True)
        2. (self.logits * one_hot).sum().backward(retain_graph=True)
        """
        one_hot = self._encode_one_hot(ids)
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image
        self.image.requires_grad = False
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers  # list

        def forward_hook(key):
            def forward_hook_(module, input, output):
                # Save featuremaps
                a = output.detach().cpu()
                self.fmap_pool[key] = a
                del output
                del a
                torch.cuda.empty_cache()

            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                # Save the gradients correspond to the featuremaps
                a = grad_out[0].detach().cpu()
                self.grad_pool[key] = a
                torch.cuda.empty_cache()

            return backward_hook_

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(
                    module.register_full_backward_hook(backward_hook(name))
                )

    def find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def _compute_grad_weights(self, grads):
        return F.adaptive_avg_pool2d(grads, 1)

    def forward(self, image):
        self.image = image
        self.image_shape = image.shape[2:]
        return super(GradCAM, self).forward(self.image)

    def generate(self, target_layer):
        fmaps = self.find(self.fmap_pool, target_layer)
        grads = self.find(self.grad_pool, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )
        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)
        return gcam


device = torch.device("cpu")


def get_image(img_num):
    img_path = "data/images/" + img_num + ".png"
    img = imread(img_path)
    return img


def visualize_input(img_num, axarr):
    xi = get_image(img_num)
    axarr[0].imshow(utils.denormalize_image(xi))
    axarr[0].axis("off")


def visualize_layer1_activations(img_num, i, axarr):
    xi = get_image(img_num)
    xi = xi.transpose(2, 0, 1)
    xi = torch.from_numpy(xi).float()
    xi = xi.view((1, 3, 64, 64))
    bp = BackPropagation(model=model)
    gcam = GradCAM(model=model)
    target_layer = "conv1"
    target_class = 1
    _ = gcam.forward(xi)
    gcam.backward(ids=torch.tensor([[target_class]]).to(device))
    regions = gcam.generate(target_layer=target_layer)
    activation = regions.detach()
    save_gradcam(
        np.squeeze(activation),
        utils.denormalize_image(np.squeeze(xi.numpy()).transpose(1, 2, 0)),
        axarr,
        i,
    )


if __name__ == "__main__":
    # Attempts to restore from checkpoint
    print("Loading cnn...")
    model = Target()
    model, start_epoch, _ = restore_checkpoint(
        model, config("target.checkpoint"), force=True
    )

    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("target.batch_size"),
    )

    # img_list contains the ids for a sample of images from the training set
    img_list = ['01105', '00024', '03545', '05934', '03165',
                '05354', '01295', '00914', '03035', '07984',
                '09125', '05654', '05125', '05174', '04685',
                '12235', '00975', '01074', '02995', '01134']
    for i, img_num in enumerate(img_list):
        plt.clf()
        f, axarr = plt.subplots(1, 2)
        visualize_input(img_num, axarr)
        visualize_layer1_activations(img_num, i, axarr)
        plt.close()
