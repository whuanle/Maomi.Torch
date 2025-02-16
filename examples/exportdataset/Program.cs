using TorchSharp;
using static TorchSharp.torch;
using datasets = TorchSharp.torchvision.datasets;
using transforms = TorchSharp.torchvision.transforms;
using Maomi.Torch;
using TorchSharp.Modules;
using static TorchSharp.torch.distributions.transforms;
using static TorchSharp.torch.optim;

// 使用 GPU 启动
Device defaultDevice = MM.GetOpTimalDevice();
torch.set_default_device(defaultDevice);

var train_dataset = datasets.CIFAR10(root: "E:/datasets/CIFAR-10", train: true, download: true);
var val_dataset = datasets.CIFAR10(root: "E:/datasets/CIFAR-10", train: false, download: true);

var train_loader = new DataLoader(train_dataset, batchSize: 1, shuffle: true, device: defaultDevice, num_worker: 10);
var val_loader = new DataLoader(val_dataset, batchSize: 1, shuffle: false, device: defaultDevice, num_worker: 10);

var classes = new string[] {
"airplane",
"automobile",
"bird",
"cat",
"deer",
"dog",
"frog",
"horse",
"ship",
"truck"
};

var basePath = "E:/datasets/t1";

if (Directory.Exists(basePath))
{
    Directory.Delete(basePath, true);
}

Directory.CreateDirectory(basePath);
Directory.CreateDirectory(basePath + "/train");
Directory.CreateDirectory(basePath + "/test");

foreach (var item in classes)
{
    Directory.CreateDirectory(basePath + "/train" + "/" + item);
    Directory.CreateDirectory(basePath + "/test" + "/" + item);
}

int i = 0;
foreach (var item in train_loader)
{
    var (inputs, labels) = (item["data"], item["label"]);
    var classIndex = labels.item<long>();
    inputs.SaveJpeg(basePath + $"/train/{classes[classIndex]}/{i}.jpg");
    i++;
}

i = 0;
foreach (var item in val_loader)
{
    var (inputs, labels) = (item["data"], item["label"]);
    var classIndex = labels.item<long>();
    inputs.SaveJpeg(basePath + $"/test/{classes[classIndex]}/{i}.jpg");
    i++;
}