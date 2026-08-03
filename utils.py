from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform = None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = list(os.listdir(root))
        self.files = [p for p in self.files if p.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"ERROR OPENING FILE {image_path}", e)
            raise

        if self.transform:
            image = self.transform(image)

        return image

def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())


    return transforms.Compose(transform_list)


def AdaIN(content_features, style_features):
    size = content_features.size()
    style_mean, style_std = calc_mean_std(style_features)
    content_mean, content_std = calc_mean_std(content_features)
    normalized_content_feat = (content_features - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert(len(size) == 4)
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std    