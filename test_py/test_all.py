import os, sys
import random
import torch
import torchvision.transforms.functional as TF

sys.path.append(os.getcwd())

import extension
from extension import DataType

min_size = 100
max_size = 1024
def test_resize():
    x = torch.rand(3, 10, 10)
    xx = x.unsqueeze(0)
    y1 = torch.nn.functional.interpolate(xx, size=(20, 20), mode='bicubic')
    y2 = extension.resize(x, (20, 20))
    assert torch.cosine_similarity(y1.squeeze(0), y2).mean() > 0.99

def test_crop():
    for _ in range(10):
        h = random.randint(min_size, max_size)
        w = random.randint(min_size, max_size)
        x = torch.rand(3, h, w)

        y1 = TF.crop(x, 0, 0, h + 100, w + 100)
        y2 = extension.crop(x, 0, 0, h + 100, w + 100)
        assert torch.allclose(y1, y2)

        y1 = TF.crop(x, 0, 0, h - 50, w - 50)
        y2 = extension.crop(x, 0, 0, h - 50, w - 50)
        assert torch.allclose(y1, y2)

def test_center_crop():
    for _ in range(10):
        h = random.randint(min_size, max_size)
        w = random.randint(min_size, max_size)
        x = torch.rand(3, h, w)
        
        y1 = TF.center_crop(x, [h + 100, w + 100])
        y2 = extension.center_crop(x, (h + 100, w + 100))
        assert torch.allclose(y1, y2)
        y1 = TF.center_crop(x, [h - 50, w - 50])
        y2 = extension.center_crop(x, (h - 50, w - 50))
        assert torch.allclose(y1, y2)

# def test_resized_crop():
#     for _ in range(10):
#         h = random.randint(min_size, max_size)
#         w = random.randint(min_size, max_size)
#         x = torch.rand(3, h, w)
        
#         y1 = TF.resized_crop(x, 1, 1, h // 2, w // 2, [h + 100, w + 100], interpolation=TF.InterpolationMode.BICUBIC)
#         y2 = extension.resized_crop(x, 1, 1, h // 2, w // 2, (h + 100, w + 100))
#         assert torch.cosine_similarity(y1, y2).mean() > 0.99
#         y1 = TF.resized_crop(x, 1, 1, h // 2, w // 2, [h - 50, w - 50], interpolation=TF.InterpolationMode.BICUBIC)
#         y2 = extension.resized_crop(x, 1, 1, h // 2, w // 2, (h - 50, w - 50))
#         assert torch.cosine_similarity(y1, y2).mean() > 0.99

def test_normalize():
    for _ in range(10):
        h = random.randint(min_size, max_size)
        w = random.randint(min_size, max_size)
        x = torch.rand(3, h, w)
        
        mean = [random.random() for _ in range(3)]
        std = [random.random() for _ in range(3)]
        
        y1 = TF.normalize(x, mean, std)
        y2 = extension.normalize(x, mean, std, False)

        assert torch.allclose(y1, y2)

        x1 = x.clone()
        x2 = x.clone()

        y1 = TF.normalize(x1, mean, std, True)
        y2 = extension.normalize(x2, mean, std, True)

        assert torch.allclose(y1, y2)
        assert torch.allclose(x1, x2)
        assert torch.allclose(x1, y1)
        assert torch.allclose(x1, y2)
        assert torch.allclose(x2, y1)
        assert torch.allclose(x2, y2)

def test_flip():
    for _ in range(10):
        h = random.randint(min_size, max_size)
        w = random.randint(min_size, max_size)
        x = torch.rand(3, h, w)
        y1 = TF.hflip(x)
        y2 = extension.hflip(x)
        assert torch.allclose(y1, y2)
        y1 = TF.vflip(x)
        y2 = extension.vflip(x)
        assert torch.allclose(y1, y2)
    
def test_rgb_to_grayscale():
    for _ in range(10):
        h = random.randint(min_size, max_size)
        w = random.randint(min_size, max_size)
        x = torch.rand(3, h, w)
        y1 = TF.rgb_to_grayscale(x, 1)
        y2 = extension.rgb_to_grayscale(x, 1)
        assert torch.allclose(y1, y2)
        y1 = TF.rgb_to_grayscale(x, 3)
        y2 = extension.rgb_to_grayscale(x, 3)
        assert torch.allclose(y1, y2)

def test_adjust_brightness():
    for _ in range(10):
        h = random.randint(min_size, max_size)
        w = random.randint(min_size, max_size)
        x = torch.rand(3, h, w)
        p = random.random()
        y1 = TF.adjust_brightness(x, p)
        y2 = extension.adjust_brightness(x, p)
        assert torch.allclose(y1, y2)

def test_adjust_contrast():
    for _ in range(10):
        h = random.randint(min_size, max_size)
        w = random.randint(min_size, max_size)
        x = torch.rand(3, h, w)
        p = random.random()
        y1 = TF.adjust_contrast(x, p)
        y2 = extension.adjust_contrast(x, p)
        assert torch.allclose(y1, y2)
        
def test_adjust_saturation():
    for _ in range(10):
        h = random.randint(min_size, max_size)
        w = random.randint(min_size, max_size)
        x = torch.rand(3, h, w)
        p = random.random()
        y1 = TF.adjust_saturation(x, p)
        y2 = extension.adjust_saturation(x, p)
        assert torch.allclose(y1, y2)
                
def test_convert_image_dtype():
    for _ in range(10):
        h = random.randint(min_size, max_size)
        w = random.randint(min_size, max_size)
        x = torch.rand(3, h, w)
        y1 = TF.convert_image_dtype(x, torch.float32)
        y2 = extension.convert_image_dtype(x, DataType.float32)
        assert torch.allclose(y1, y2)
        