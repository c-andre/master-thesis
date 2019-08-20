
import PIL.Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseImage:
    plt.ion()

    def __init__(self, data):
        self.data = data

    @classmethod
    def from_file(cls, file):
        return cls(PIL.Image.open(file))

    def show(self, image=None, title=None):
        if image is None:
            image = self.data

        plt.figure()
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.02)


    def save(self, file):
        self.data.save(file)


class Image(BaseImage):

    def __init__(self, data):
        super(Image, self).__init__(data)

    @classmethod
    def from_tensor(cls, tensor):
        image = tensor.cpu().clone().squeeze(0)
        return cls(transforms.ToPILImage()(image))

    def to_tensor(self):
        tensor = transforms.ToTensor()(self.data)
        return tensor.unsqueeze(0).to(device, torch.float)

    def resize(self, size):
        self.data = transforms.Resize(size)(self.data)
        return self
