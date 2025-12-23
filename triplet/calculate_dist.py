import torch
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image
from triplet.triplet_model import triplet

device = "mps" if torch.backends.mps.is_available() else "cpu"

transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def transform(path):
    img = Image.open(path).convert("RGB")
    img = transformer(img)
    img = img.unsqueeze(0).to(device)
    vec = triplet.predict(img)
    return vec

dir_link = "/Users/maxkucher/PycharmProjects/face_2_face/faces"
list_dir = os.listdir(dir_link)

embeddings = []
for img in list_dir:
    full_path = os.path.join(dir_link, img)
    img_tensor = transform(full_path)
    embeddings.append(img_tensor)

embeddings = torch.cat(embeddings, dim=0)
orig_vector = embeddings.mean(dim=0, keepdim=True)
orig_vector = F.normalize(orig_vector, dim=1)

def calculate_dist(vector_1, vector_2):
    dist = torch.norm(vector_1 - vector_2)
    return dist
    # if dist < threshold:
    #     return 1
    # else:
    #     return 0




