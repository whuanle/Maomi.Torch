from PIL import Image
from torchvision import transforms
import torch
from torchvision import models
from torchvision.transforms import ToPILImage

# if torch.cuda.is_available():
#     print("当前设备支持 GPU")
#     device = torch.device('cuda')
#     # 使用 GPU 启动
#     torch.set_default_device(device)
#     current_device = torch.cuda.current_device()
#     print(f"绑定的 GPU 为：{current_device}")
# else:
#     # 不支持 GPU，使用 CPU 启动
#     device = torch.device('cpu')
#     torch.set_default_device(device)
#
# default_device = torch.get_default_device()
# print(f"当前正在使用 {default_device}")

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

img = Image.open("bobby.jpg")
img_t = preprocess(img)

show = ToPILImage()
show(img_t).show()

batch_t = torch.unsqueeze(img_t, 0)

resnet101 = models.resnet101(pretrained=True)
# resnet = resnet101.to(default_device)
resnet101.eval()

out = resnet101(batch_t)

with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

_, indices = torch.sort(out, descending=True)
for idx in indices[0][:5]:
    print(labels[idx])
    print(percentage[idx].item())
