import torchvision.models.video as models

import torch
from torch import nn
import torch.nn.functional as F



class TeacherModel(nn.Module):
    def __init__(self, out_channels=50):
        super().__init__()

        self.model = models.r3d_18(weights="KINETICS400_V1")
        self.model.fc = nn.Linear(512, out_channels)

    def forward(self, x):
        out = self.model(x)
        return out

    def predict(self, x):
        self.eval()

        if len(x.shape) == 4:
            x = x.squeeze(3)

        with torch.no_grad():
            out = self.model(x)

        out = F.softmax(out, dim=1)
        res = torch.argmax(out, dim=1).item()
        return res

device = "mps" if torch.backends.mps.is_available() else "cpu"
link = "conv_3d_model/teacher_model.pt"
model = TeacherModel()
model.load_state_dict(torch.load(link, map_location=device))
print("All keys matched successfully")

