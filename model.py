import torchvision.models.video as video_models
import torchvision.models as models

import torch
from torch import nn
import torch.nn.functional as F

class TeacherNet(nn.Module):
    def __init__(self, out_channels=50):
        super().__init__()

        self.model = video_models.r3d_18(weights="KINETICS400_V1")
        self.model.fc = nn.Linear(512, out_channels)

    def forward(self, x):
        out = self.model(x)
        # out = F.softmax(out, dim=1)
        return out 

    def predict(self, x):
        self.eval()

        if x.shape == 4:
            x = x.unsqueeze(0)

        with torch.no_grad(): 
            out = self.forward(x)
        out = F.softmax(out, dim=1)
        res = torch.argmax(out, dim=1).item()
        return res 
    


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.attn = nn.Linear(dim, 1)
    
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


class StudentNet(nn.Module):
    def __init__(self, num_layers=1, input_size=512, hidden_size=256, num_classes=50, bidirectional=True):
        super().__init__()

        body = models.resnet34(weights="DEFAULT")
        self.cnn = nn.Sequential(*list(body.children())[:-2])
        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.lstm = nn.LSTM(
            input_size = input_size, 
            hidden_size = hidden_size,
            num_layers = num_layers, 
            batch_first = True,
            bidirectional = True
        )

        lstm_out = hidden_size * 2 if bidirectional else hidden_size
        
        self.temporal_attn = TemporalAttention(lstm_out)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, F, C, H, W = x.shape
        x = x.reshape(B*F, C, H, W)

        out = self.cnn(x)
        out = self.aap(out)
        out = self.flatten(out)

        out = out.reshape(B, F, -1)
        lstm_out, (_, _) = self.lstm(out)
        out = self.temporal_attn(lstm_out)
        
        out = self.classifier(out)

        return out

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
        out = torch.softmax(out, dim=1)
        out = torch.argmax(out, dim=1)
        return out.item()


# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
link = "./student_model.pt"
model = StudentNet()
model.load_state_dict(torch.load(link, map_location=device))
# model.to(device)
print("All keys matched successfully")

# tensor = torch.randn(1, 3, 16, 128, 128).to(device)
# pred = model.predict(tensor)
# print(pred)





