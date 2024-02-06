import os,sys
import torch
import torchvision.transforms.functional as TF

sys.path.append(os.getcwd())

import extension

def test_resize():
    x = torch.rand(3, 10, 10)
    xx = x.unsqueeze(0)
    y1 = torch.nn.functional.interpolate(xx, size=(20, 20), mode='bicubic')
    y2 = extension.resize(x, (20, 20))
    assert torch.cosine_similarity(y1.squeeze(0), y2).mean() > 0.99

def test_crop():
    x = torch.rand(3, 9, 9)
    y1 = TF.crop(x, 1, 1, 4, 4)
    y2 = extension.crop(x, 1, 1, 4, 4)
    assert torch.allclose(y1, y2)

def test_center_crop():
    x = torch.rand(3, 9, 9)
    y1 = TF.center_crop(x, [11, 11])
    y2 = extension.center_crop(x, (11, 11))
    assert torch.allclose(y1, y2)
    y1 = TF.center_crop(x, [4, 4])
    y2 = extension.center_crop(x, (4, 4))
    assert torch.allclose(y1, y2)

def test_resized_crop():
    x = torch.rand(3, 9, 9)
    y1 = TF.resized_crop(x, 1, 1, 4, 4, [11, 11])
    y2 = extension.resized_crop(x, 1, 1, 4, 4, [11, 11])
    assert torch.allclose(y1, y2)
    y1 = TF.resized_crop(x, 1, 1, 4, 4, [4, 4])
    y2 = extension.resized_crop(x, 1, 1, 4, 4, (4, 4))
    assert torch.allclose(y1, y2)

def test_normalize():
    x = torch.rand(3, 300, 300)

    y1 = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    y2 = extension.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], False)

    assert torch.allclose(y1, y2)

    x1 = x.clone()
    x2 = x.clone()

    y1 = TF.normalize(x1, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True)
    y2 = extension.normalize(x2, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True)

    assert torch.allclose(y1, y2)
    assert torch.allclose(x1, x2)
    assert torch.allclose(x1, y1)
    assert torch.allclose(x1, y2)
    assert torch.allclose(x2, y1)
    assert torch.allclose(x2, y2)