from torchvision import models
import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        body = models.resnet18(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(body.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = F.normalize(x, dim=1)
        return x

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            x = self.forward(x)
        return x

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = Net()
model.load_state_dict(torch.load("/Users/maxkucher/PycharmProjects/face_2_face/triplet/triplet_model.pt", map_location=device))
model.to(device)
print(f"All keys matched successfully of device: {device}!")