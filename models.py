import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from constructor import ResNetBlock, SkipBlock, SkipConv, ResBlock


device = 'mps' if torch.backends.mps.is_available() else 'cpu'


# ------------------------------------------------------------------------------------------------------------------------------------
class AgeModel(nn.Module):
    def __init__(self, outputs=5):
        super().__init__()
        # 3, 224, 224
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding = 3)
        self.pool1 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding = 3)
        self.pool2 = nn.MaxPool2d(stride=2, kernel_size=2)
        # 64, 56, 56
        self.block2 = ResNetBlock(64, 128, bnorm = True)
        self.pool2 = nn.MaxPool2d(stride=2, kernel_size=2)
        
        # 128, 28, 28
        self.block3 = ResNetBlock(128, 256, bnorm = True)
        self.pool3 = nn.MaxPool2d(stride=2, kernel_size=2)
        # 256, 14, 14
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size = 3, padding = 1)
        self.pool4 = nn.MaxPool2d(stride=2, kernel_size=2)
        # 512, 7, 7

        
        self.conv5 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
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

        with torch.no_grad():
            res = self.forward(x)
        pred = torch.argmax(res, dim=1)

        return pred.item()

age_model_link = "./weights/age_model.pt"
age_model = AgeModel()
age_model.load_state_dict(torch.load(age_model_link, map_location=device))


# # ------------------------------------------------------------------------------------------------

class EmotionModel(nn.Module):
    def __init__(self, outputs = 4):
        super().__init__()
        
        # 3, 224, 224         
        self.block1 = SkipBlock(3, 32, pool = True)
        # 32, 112, 112
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64, 56, 56      
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 128, 28, 28
        self.block4 = SkipBlock(128, 256, pool = True)
        # 256, 14, 14
        self.block5 = SkipBlock(256, 512, pool = True)
        # 512, 7, 7
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size = 4, stride=4)
        

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
        
        with torch.no_grad():
            
            res = self.forward(x)
            
        fin_res = torch.argmax(res, dim = 1)
            
        return fin_res.item()
                

emotion_model_link = "./weights/emotion_model.pt"
emotion_model = EmotionModel()
emotion_model.load_state_dict(torch.load(emotion_model_link, map_location=device))

# # ------------------------------------------------------------------------------------------------

class GenderModel(nn.Module):
    def __init__(self, outputs = 1):
        super().__init__()
        # 3, 224, 224
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # 32, 112, 112
        self.block2 = SkipBlock(32, 64, pool = False)
        # 64, 56, 56
        self.block3 = SkipBlock(64, 128, pool = False)
        # 128, 28, 28

        # self.block4 = SkipConBlock(128, 256, bnorm = False, pool = False)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size = 3, padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size = 3, padding = 1)
        self.pool4_1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # 256, 7, 7
        self.block5 = SkipBlock(512, 512, pool = False)
        # 512, 3, 3
        self.conv5 = nn.Conv2d(512, 1024, kernel_size = 3, padding = 1)
        self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2)
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
        self.eval()

        with torch.no_grad():
            res = self.forward(x)
            res = (res >= 0.5).long()

        return res.item()

gender_model_link = "./weights/gender_model.pt"
gender_model = GenderModel()
gender_model.load_state_dict(torch.load(gender_model_link, map_location=device))
        
        

# # ------------------------------------------------------------------------------------------------


class RaceModel(nn.Module):
    def __init__(self, outputs = 3):
        super().__init__()
        
        # 3, 224, 224         
        self.block1 = SkipConv(3, 32, pool = True)
        # 32, 112, 112
        self.block2 = SkipConv(32, 64, pool = True)
        # 64, 56, 56
        # self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        # self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.block3 = SkipConv(64, 128, pool = True)
        
        # 128, 28, 29
        self.conv4 = nn.Conv2d(128, 256, kernel_size = 3, padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pool4_1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # 256, 7, 7
        self.block5 = SkipConv(256, 512, pool = True)
        
        # self.conv5 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        # self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # 512, 3, 3
        # self.block6 = SkipConv(512, 512, pool = True)
        self.conv6 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.pool6 = nn.MaxPool2d(kernel_size = 2, stride = 2)
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
        with torch.no_grad():
            
            res = self.forward(x)
            
        fin_res = torch.argmax(res, dim = 1)
            
        return fin_res.item()


    

race_model_link = "./weights/race_model.pt"
race_model = RaceModel()
race_model.load_state_dict(torch.load(race_model_link, map_location=device))


if __name__ == "__main__":

    try:
        tensor = torch.rand(1, 3, 224, 224)
        age_model.predict(tensor)
        emotion_model.predict(tensor)
        gender_model.predict(tensor)
        race_model.predict(tensor)

        print("All models are correct!")
    except Exception as e:
        print(f"Some problem with custom models tensor sizes: {e}. Check tools.py or models.py for more info.")


