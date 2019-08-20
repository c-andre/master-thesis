import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import click
import warnings
import math
import numpy as np

def patches(x, patch_width, stride=1):
    patches = F.unfold(x, patch_width, padding=1, stride=stride).squeeze(0).t()
    norms = torch.norm(patches, p=2, dim=1)
    return patches, norms

def mapping(input_features, style_features, width=3, eps=1e-10):
    mapping = []

    for input, style in zip(input_features, style_features):

        # extract patches of size (width x width), compute their l2 norm
        input_patches, input_norms = patches(input, width)
        style_patches, style_norms = patches(style, width)

        # compute the best mapping according to the cosine similarity
        num = torch.mm(input_patches, style_patches.t())
        den = input_norms.view(1, -1) * style_norms.view(-1, 1)
        mapping.append(torch.argmax(num / den.clamp(min=eps), 0))

    return mapping

# Align the style features with the content using the provided mapping
def align(style_features, mapping):
    aligned = []
    for style, maps in zip(style_features, mapping):
        style = style.reshape(style.size(1), -1)
        style = style[:, maps]
        aligned.append(style)
    return aligned

# Propagate the reference mapping to another layer
def propagate_mapping(mapping, dim_ref, dim_l):
    h_ref, w_ref = dim_ref
    h_l, w_l = dim_l

    # establish correspondences between the indices of patches and their (i,j)
    # coordinates
    I_l, J_l = torch.meshgrid([torch.arange(0, h_l), torch.arange(0, w_l)])
    I_l, J_l = I_l.flatten(), J_l.flatten()
    I_ref, J_ref = torch.meshgrid([torch.arange(0, h_ref), torch.arange(0, w_ref)])
    I_ref, J_ref = I_ref.flatten(), J_ref.flatten()
    index_l = torch.arange(0, h_l*w_l).reshape((h_l,w_l))
    index_ref = torch.arange(0, h_ref*w_ref).reshape((h_ref,w_ref))

    # propagated mapping
    mapping_l = torch.zeros(h_l*w_l).long()

    for k_l, (i_l, j_l) in enumerate(zip(I_l, J_l)):

        # get center of the corresponding patch in the reference layer
        # clamp the values to ensure that we stay within the dimensions
        # of the feature map
        i_ref =  torch.clamp(torch.round(i_l.double() / h_l * h_ref), 0, h_ref-1).long()
        j_ref =  torch.clamp(torch.round(j_l.double() / w_l * w_ref), 0, w_ref-1).long()

        # obtain the best matching style patch in the reference layer
        k_ref = mapping[index_ref[i_ref,j_ref]]
        i_ref, j_ref = I_ref[k_ref], J_ref[k_ref]

        # get back to the layer l
        i_l = torch.clamp(torch.round(i_ref.double() * h_l / h_ref), 0, h_l-1).long()
        j_l = torch.clamp(torch.round(j_ref.double() * w_l / w_ref), 0, w_l-1).long()
        # assign the obtained style patch to the current content patch
        mapping_l[k_l] = index_l[i_l,j_l]
    return mapping_l

def refined_mapping(map, style_features, mask, width=5):

    c, h, w = style_features.size()
    style_features = style_features.reshape((c, h*w)).t()

    # store the (i,j) coordinates for every pixel of the style features
    I, J = torch.meshgrid([torch.arange(0, h), torch.arange(0, w)])
    I = I.flatten()
    J = J.flatten()

    index = torch.arange(0, h*w).reshape((h,w))

    # store the offsets defining the neighbourhood of a pixel
    offsets = torch.arange(- (width // 2), width // 2 + 1)
    DI, DJ = torch.meshgrid([offsets, offsets])

    # for every pixel in the style feature map
    for k, (i, j) in enumerate(zip(I, J)):

        # ignore masked out pixels
        if mask[k] < 0.5: continue

        neighbours = []

        # for every neihbouring pixel
        for di, dj in zip(DI.flatten(), DJ.flatten()):
            if 0 <= i+di < h and 0 <= j+dj < w:
                # retrieve the corresponding style patch p
                p = map[index[i+di,j+dj]]
                # apply the opposite offset to the style patch
                neighbours.append(index[I[p]-di, J[p]-dj].item())

        # select the candidate closest to the other neighbours on average,
        # modify the mapping accordingly
        min_dist = float('inf')
        for c in set(neighbours):
            dist = 0
            for n in neighbours:
                dist += ((style_features[c] - style_features[n]) ** 2).sum()
            if dist < min_dist:
                min_dist = dist
                map[k] = c
    return map
