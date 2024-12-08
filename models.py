import torch
from holoviews.plotting.bokeh.util import multi_polygons_data
from torch import nn
import torch.nn.functional as F
from constructor import SkipConBlock, SkipBlock

device = 'mps' if torch.backends.mps.is_available() else 'cpu'


# ------------------------------------------------------------------------------------------------------------------------------------
class SmallModel(nn.Module):
    def __init__(self, outputs=5):
        super().__init__()
        # 3, 224, 224
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=3)
        self.pool1 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=3)
        self.pool2 = nn.MaxPool2d(stride=2, kernel_size=2)
        # 64, 56, 56
        self.block2 = SkipConBlock(64, 128, bnorm=True)
        self.pool2 = nn.MaxPool2d(stride=2, kernel_size=2)

        # 128, 28, 28
        self.block3 = SkipConBlock(128, 256, bnorm=True)
        self.pool3 = nn.MaxPool2d(stride=2, kernel_size=2)
        # 256, 14, 14

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(stride=2, kernel_size=2)
        # 512, 7, 7

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(512, 128)
        self.bnorm1_1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.25)

        self.linear2 = nn.Linear(128, 32)
        self.drop2 = nn.Dropout(0.25)

        self.linear3 = nn.Linear(32, outputs)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.pool2(out)
        out = F.relu(out)

        out = self.block2(out)
        out = self.pool2(out)

        out = self.block3(out)
        out = self.pool3(out)
        out = self.pool3(out)

        out = self.conv4(out)
        out = F.relu(out)
        out = self.pool4(out)

        out = self.conv5(out)
        out = F.relu(out)
        out = self.pool5(out)

        # out = self.av_pooling(out)
        # out = out.view(out.size(0), -1)

        out = self.flatten(out)

        out = self.linear1(out)
        out = self.bnorm1_1(out)
        out = F.relu(out)
        # out = self.drop1(out)

        out = self.linear2(out)
        out = F.relu(out)
        out = self.drop2(out)

        res = self.linear3(out)

        # res = self.linear3(out)

        return res

    def predict(self, x):
        self.eval()

        new_x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            t_res = self.forward(new_x)
            res = torch.softmax(t_res, dim=1)
            fin_res = torch.argmax(res, dim=1)

        return fin_res


def age_model(data):
    model = SmallModel().to(device)
    model.load_state_dict(torch.load("weights/final_age_model.pt", map_location=device))
    model.eval()

    t_res = model.predict(data)

    fin_res = t_res.item()

    return fin_res

# try
# data = torch.rand(1, 3, 224, 224).to(device)
# res = age_model(data)

# check
# model = SmallModel()
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# loaded_state_dict = torch.load("weights/new_age_model.pt", map_location=device)
# print("Keys in loaded state_dict:", loaded_state_dict.keys())
# print("Keys in SmallModel state_dict:", model.state_dict().keys())


# ------------------------------------------------------------------------------------------------

class SmallEmotionCLassifier(nn.Module):
    def __init__(self, outputs=4):
        super().__init__()

        # 3, 224, 224
        self.block1 = SkipBlock(3, 32, pool=True)
        # 32, 112, 112
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64, 56, 56
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 128, 28, 28
        self.block4 = SkipBlock(128, 256, pool=True)
        # 256, 14, 14
        self.block5 = SkipBlock(256, 512, pool=True)
        # 512, 7, 7
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(512, 128)
        self.drop = nn.Dropout(0.2)
        self.linear2 = nn.Linear(128, outputs)

    def forward(self, x):
        out = self.block1(x)

        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = F.relu(out)
        out = self.pool3(out)

        out = self.block4(out)
        out = self.block5(out)

        out = self.conv6(out)
        out = F.relu(out)
        out = self.pool6(out)

        out = self.flatten(out)

        out = self.linear1(out)
        out = F.relu(out)
        out = self.drop(out)

        res = self.linear2(out)

        return res

    def predict(self, x):
        self.eval()

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            t_res = self.forward(x)
            res = torch.softmax(t_res, dim=1)
            fin_res = torch.argmax(res, dim=1)

        return fin_res


def emotion_model(data):
    model = SmallEmotionCLassifier().to(device)
    model.load_state_dict(torch.load("weights/compact_emotion_model.pt", map_location=device))

    res = model.predict(data)

    fin_res = res.item()
    return fin_res

# ------------------------------------------------------------------------------------------------

class GenderDetector(nn.Module):
    def __init__(self, outputs=1):
        super().__init__()
        # 3, 224, 224
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 32, 112, 112
        self.block2 = SkipBlock(32, 64, pool=False)
        # 64, 56, 56
        self.block3 = SkipBlock(64, 128, pool=False)
        # 128, 28, 28

        # self.block4 = SkipConBlock(128, 256, bnorm = False, pool = False)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool4_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 256, 7, 7
        self.block5 = SkipBlock(512, 512, pool=False)
        # 512, 3, 3
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 512, 1, 1

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(1024, 512)
        self.bnorm1_1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)

        self.linear2 = nn.Linear(512, 32)
        self.linear3 = nn.Linear(32, outputs)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.pool1(out)

        out = self.block2(out)
        ut = F.relu(out)
        out = self.pool1(out)

        out = self.block3(out)
        out = F.relu(out)
        out = self.pool1(out)

        out = self.conv4(out)
        out = F.relu(out)
        out = self.pool4(out)

        out = self.conv4_1(out)
        out = F.relu(out)
        out = self.pool4_1(out)

        out = self.block5(out)
        out = F.relu(out)
        # out = self.pool1(out)

        out = self.conv5(out)
        out = self.pool5(out)
        out = F.relu(out)

        out = self.flatten(out)

        out = self.linear1(out)
        out = F.relu(out)
        out = self.bnorm1_1(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        res = F.sigmoid(out)

        return res

    def predict(self, x):
        data = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        self.eval()

        with torch.no_grad():
            res = self.forward(data)

        return res


def gender_model(data):

    fin_res = None
    model = GenderDetector().to(device)

    model.load_state_dict(torch.load("weights/gender_model.pt", map_location=device))

    model.eval()

    res = model.predict(data)

    if res >= 0.5:
        fin_res = 1
    elif res < 0.5:
        fin_res = 0

    return fin_res


# ------------------------------------------------------------------------------------------------

class RaceNetwork(nn.Module):
    def __init__(self, outputs=3):
        super().__init__()

        # 3, 224, 224
        self.block1 = SkipBlock(3, 32, pool=True)
        # 32, 112, 112
        self.block2 = SkipBlock(32, 64, pool=True)
        # 64, 56, 56
        # self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        # self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.block3 = SkipBlock(64, 128, pool=True)

        # 128, 28, 29
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 256, 7, 7
        self.block5 = SkipBlock(256, 512, pool=True)

        # self.conv5 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        # self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # 512, 3, 3
        # self.block6 = SkipConv(512, 512, pool = True)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 512, 1, 1

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(512, 256)
        self.drop1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(256, 32)

        self.linear3 = nn.Linear(32, outputs)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)

        out = self.block3(out)

        # out = self.conv3(out)
        # out = F.relu(out)
        # out = self.pool3(out)

        out = self.conv4(out)
        out = F.relu(out)
        out = self.pool4(out)
        out = self.pool4_1(out)

        out = self.block5(out)

        # out = self.conv5(out)
        # out = F.relu(out)
        # out = self.pool5(out)

        # out = self.block6(out)

        out = self.conv6(out)
        out = F.relu(out)
        out = self.pool6(out)

        out = self.flatten(out)

        out = self.linear1(out)
        out = self.drop1(out)
        out = F.relu(out)

        out = self.linear2(out)
        out = F.relu(out)
        res = self.linear3(out)
        # res = torch.softmax(out)

        return res

    def predict(self, x):
        self.eval()

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            t_res = self.forward(x)

            res = torch.softmax(t_res, dim=1)
            fin_res = torch.argmax(res, dim=1)

        return fin_res

def race_model(data):
    model = RaceNetwork().to(device)
    model.load_state_dict(torch.load("weights/races_model.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        res = model.predict(data)

    return res


if __name__ == "__main__":

    try:
        tensor = torch.rand(3, 224, 224).to(device)
        age_model(tensor)

        tensor = torch.rand(3, 224, 224).to(device)
        emotion_model(tensor)

        tensor = torch.rand(3, 224, 224).to(device)
        gender_model(tensor)

        tensor = torch.rand(3, 224, 224).to(device)
        race_model(tensor)

        print("All models are correct!")
    except Exception as e:
        print(f"Some problem with custom models tensor sizes: {e}. Check tools.py or models.py for more info.")



