from torchvision import transforms

class Preprocessor:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    def preprocess(self, crop):
        return self.transforms(crop)