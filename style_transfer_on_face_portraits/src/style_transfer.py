from helpers.image import Image

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

from mapping import mapping, refined_mapping, align, propagate_mapping

# use the GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# style transfer for head portraits variant
use_gain_maps = True

def gain_map(content, style, eps=1e-04, g_min=0.7, g_max=5.0):
    G = style / (content + eps)
    return torch.clamp(G, g_min, g_max)


# allows the extraction of feature maps in the forward pass
class Hook(nn.Module):
    features = None

    def __init__(self, m, transform=lambda x: x):
        self.hook = m.register_forward_hook(self.hook_fn)
        self.transform = transform

    def hook_fn(self, module, input, output):
        self.features = self.transform(output)

    def close(self):
        self.hook.remove()

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize the image
        return (img - self.mean) / self.std

class Net(nn.Module):
    # normalization applied to images trained by VGG
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization_layer = Normalization(mean, std).to(device)

    # a pretrained (on ImageNet) vgg19 classifier is used for feature extraction
    vgg = models.vgg19(pretrained=True).features.to(device).eval()

    # remove the deeper, unused layers from the network to avoid having to
    # perform the complete forward pass
    def trim(self, vgg, last_layer):
        model = nn.Sequential(Net.normalization_layer)

        for idx, layer in enumerate(list(vgg.children())[:last_layer+1]):
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)

            model.add_module(str(idx), layer)
        return model

    def __init__(self, content_layers=[], style_layers=[], mask=None):
        super(Net, self).__init__()

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.mask = mask

        self.model = self.trim(Net.vgg, max(self.content_layers+self.style_layers))

        self.content_hook = [Hook(list(self.model.children())[idx]) for idx in content_layers]
        self.style_hook = [Hook(list(self.model.children())[idx]) for idx in style_layers]

    def forward(self, image, content_features=None, style_features=None, phase=0):
        self.content_features = []
        self.style_features = []
        self.content_loss = 0
        self.style_loss = 0

        self.model(image)

        self.content_features = [h.features for h in self.content_hook]
        self.style_features = [h.features for h in self.style_hook]

        # compute the content and style losses
        if content_features is not None:
            for c, d in zip(content_features,self.content_features):
                mask = self.mask

                # change the mask resolution if needed
                if(self.mask.shape[2:4] != c.shape[2:4]):
                    mask = nn.Upsample(size=c.shape[2:4], mode='nearest')(self.mask)

                self.content_loss += square_loss(mask * c, mask * d)

        if style_features is not None:
            for a, b in zip(style_features, self.style_features):
                mask = self.mask
                s = b

                # change the mask resolution
                if(self.mask.shape[2:4] != s.shape[2:4]):
                    mask = nn.Upsample(size=s.shape[2:4], mode='nearest')(self.mask)

                b = b * mask
                b = b.view(b.size(1), -1)
                a = a * mask.view(1, -1)

                self.style_loss += F.mse_loss(gram(a), gram(b)) / (s.shape[2]*s.shape[3])

        return self

def square_loss(a,b):
    return sum(list(map(lambda x: F.mse_loss(x[0], x[1]), list(zip(a,b)))))

# sum of the variations between neighboring pixels, a measure of noise
# this loss helps obtain smoother results
def total_variation_loss(img):
    return torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
           torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

def gram(x):
    return torch.mm(x, x.t())

def run_style_transfer(
    content_img, style_img, input_img, first_pass_img, style_aligned_img,
    mask, learning_rate, content_layers, style_layers, n_iter,
    style_weight=0.007, content_weight=1.0, phase=0, pass_=1):

    # extract content features
    content_features = Net(content_layers=content_layers)(content_img, phase=phase).content_features
    style_aligned_features = Net(content_layers=content_layers, mask=mask)(style_aligned_img, phase=phase).content_features

    # modify the content features through the use of gain maps (style transfer
    # for head portraits)
    if use_gain_maps:
        for i, (c, s) in enumerate(zip(content_features, style_aligned_features)):
            content_features[i] = c * gain_map(c, s)

    # extract style features
    style_features   = Net(style_layers=style_layers, mask=mask)(style_img, phase=phase).style_features
    input_features = Net(style_layers=style_layers, mask=mask)(first_pass_img, phase=phase).style_features

    # first pass
    if pass_ == 1:
        maps = mapping(input_features, style_features)
        modified_style_features = align(style_features, maps)

    # second pass
    else:
        # index of the reference layer
        ref = 2
        # determine the matching between content and style patches
        map = mapping([input_features[ref]], [style_features[ref]])[0]
        mask = nn.Upsample(size=style_features[ref].shape[2:4], mode='nearest')(mask)

        # make the mapping more robust
        map = refined_mapping(map, style_features[ref][0], mask.reshape(-1))

        # propagate the mapping obtained at the reference layer to other style layers
        mappings = [propagate_mapping(map, style_features[ref].shape[2:4], sf.shape[2:4]) for sf in style_features]

        # align the style features based on the mapping
        modified_style_features = align(style_features, mappings)

    net = Net(content_layers=content_layers, style_layers=style_layers, mask=mask)
    features = {}

    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=learning_rate)

    run = [0]
    while run[0] <= n_iter:

        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model = net(input_img, content_features=content_features, \
                        style_features=modified_style_features, phase=phase)

            content_score = model.content_loss
            style_score = model.style_loss

            content_score = content_weight/len(content_layers) * content_score
            style_score = style_weight/len(style_layers) * style_score

            tv_loss = 0.000001 * total_variation_loss(input_img)

            loss = content_score + style_score + tv_loss
            loss.backward(retain_graph=True)


            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f} TV Loss: {:4f}'.format(
                    style_score, content_score, tv_loss, loss))
                Image.from_tensor(input_img).save("./frames/frame-{}-{}.png".format(phase,run[0]))

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img


@click.command()
@click.option('--content')
@click.option('--style')
@click.option('--mask')
@click.option('--dev', type=click.Choice(['cpu','gpu']))
@click.option('--content_layers', default=[20])
@click.option('--style_layers', default=[1,6,11,20])
@click.option('--content_weight', type=float, default=10.0)
@click.option('--style_weight', type=float, default=10.0)
@click.option('--n_iter', type=int, default=50)
@click.option('--learning_rate', type=float, default=1.0)
@click.option('--resolution', type=int, default=256)
@click.option('--input')
@click.option('--style_aligned')
@click.option('--pass_', type=int, default=1)
def entry(content, style, mask, dev, content_layers,  \
          style_layers, style_weight, content_weight, \
          learning_rate, n_iter, resolution, input, style_aligned, pass_):

    main(content, style, mask, dev, content_layers,  \
         style_layers, style_weight, content_weight, \
         learning_rate, n_iter, resolution, input, style_aligned, pass_)

def main(content, style, mask, dev, content_layers,  \
         style_layers, style_weight, content_weight, \
         learning_rate, n_iter, resolution, input, style_aligned, pass_):

    # gpu by default
    global device, size
    resolutions = [resolution]

    content_img = Image.from_file(content).resize(resolutions[0]).to_tensor()

    mask = Image.from_file(mask).resize(resolutions[0])
    mask.data.convert('1')

    tensor = transforms.ToTensor()(transforms.Grayscale(num_output_channels=1)(mask.data))
    mask = tensor.unsqueeze(0).to(device, torch.float)

    # if this is the first pass, input should be the content image
    first_pass_img = Image.from_file(input).resize(content_img.shape[2:4]).to_tensor()
    input_img = Image.from_file(content).resize(resolutions[0]).to_tensor()

    phase=0
    for size in resolutions:

        content_img = Image.from_file(content).resize(size).to_tensor()
        size = content_img.shape[2:4]

        if phase > 0:
            input_img = Image.from_tensor(output).resize(size).to_tensor()

        style_img = Image.from_file(style).resize(size).to_tensor()
        style_aligned_img = Image.from_file(style_aligned).resize(size).to_tensor()

        modified_input = content_img

        modified_input[0]= content_img[0] * gain_map(content_img[0], style_aligned_img[0])

        Image.from_tensor(modified_input).save("modified_input.png")

        output = run_style_transfer(content_img, style_img, input_img, \
            first_pass_img, style_aligned_img,                         \
            mask, content_weight=content_weight,                       \
            content_layers=content_layers, style_layers=style_layers,  \
            phase=phase, style_weight=style_weight,                    \
            n_iter=n_iter, learning_rate=learning_rate, pass_=pass_)

        Image.from_tensor(output).save("./frames/output-{}.png".format(phase))
        phase += 1

if __name__ == "__main__":
    entry()
