import cv2
import os
import torch


def get_pass(file_name):
        if file_name.endswith('.txt'):
                full_path = os.path.abspath(file_name)
                if os.path.isfile(full_path):
                        with open(full_path, 'r') as file:
                                data = file.read()
                        return int(data)

print(get_pass("password.txt"))



def return_face(data, device):
        face = cv2.resize(data, (224, 224))

        t_face = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1).to(device)


        fin_face = t_face / 255
        return fin_face

