import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output


class SimpleNet(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleNet, self).__init__()

        # Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3, out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 , self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=128 * 4 * 4, out_features=3)

    def forward(self, input_):
        output = self.net(input_)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


def evaluate_model(img, model):
    image = Image.open(img).convert('RGB')

    transformation = transforms.Compose([transforms.Resize((128, 128)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

                                         ])
    image_tensor = transformation(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    inp = Variable(image_tensor)

    output = model(inp)
    result = output.argmax().item()

    return result


if __name__ == '__main__':
    checkpoint = torch.load('model_4.model')
    model = SimpleNet(3)
    model.load_state_dict(checkpoint)
    model.eval()

    res = evaluate_model('path_to_img', model)
