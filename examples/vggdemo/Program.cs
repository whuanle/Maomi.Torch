using TorchSharp;
using static TorchSharp.torch;
using datasets = TorchSharp.torchvision.datasets;
using transforms = TorchSharp.torchvision.transforms;
using Maomi.Torch;
using TorchSharp.Modules;
using static TorchSharp.torch.distributions.transforms;
using static TorchSharp.torch.optim;
using static SkiaSharp.SKImageFilter;

Device defaultDevice = MM.GetOpTimalDevice();
torch.set_default_device(defaultDevice);

Console.WriteLine($"当前正在使用 {defaultDevice}");

// 数据预处理
var transform = transforms.Compose([
    transforms.Resize(32, 32),
    transforms.ConvertImageDtype( ScalarType.Float32),
   MM.transforms.ReshapeTransform(new long[]{ 1,3,32,32}),
    transforms.Normalize(means: new double[] { 0.485, 0.456, 0.406 }, stdevs: new double[] { 0.229, 0.224, 0.225 }),
    MM.transforms.ReshapeTransform(new long[]{ 3,32,32})
]);

// 加载训练和验证数据

var train_dataset = datasets.CIFAR10(root: "E:/datasets/CIFAR-10", train: true, download: true, target_transform: transform);
var val_dataset = datasets.CIFAR10(root: "E:/datasets/CIFAR-10", train: false, download: true, target_transform: transform);

//var train_dataset = MM.Datasets.ImageFolder(root: "E:/datasets/t1/train", target_transform: transform);
//var val_dataset = MM.Datasets.ImageFolder(root: "E:/datasets/t1/test", target_transform: transform);

var train_loader = new DataLoader(train_dataset, batchSize: 1024, shuffle: true, device: defaultDevice, num_worker: 10);
var val_loader = new DataLoader(val_dataset, batchSize: 1024, shuffle: false, device: defaultDevice, num_worker: 10);

var model = torchvision.models.vgg16(num_classes: 10);
model.to(device: defaultDevice);

// 设置损失函数和优化器
var criterion = nn.CrossEntropyLoss();
var optimizer = optim.SGD(model.parameters(), learningRate: 0.001, momentum: 0.9);


#region 训练代码，可注释

int num_epochs = 150;

for (int epoch = 0; epoch < num_epochs; epoch++)
{
    model.train();
    double running_loss = 0.0;
    int i = 0;
    foreach (var item in train_loader)
    {
        var (inputs, labels) = (item["data"], item["label"]);
        var inputs_device = inputs.to(defaultDevice);
        var labels_device = labels.to(defaultDevice);

        optimizer.zero_grad();
        var outputs = model.call(inputs_device);
        var loss = criterion.call(outputs, labels_device);
        loss.backward();
        optimizer.step();

        running_loss += loss.item<float>() * inputs.size(0);
        Console.WriteLine($"[{epoch}/{num_epochs}][{i % train_loader.Count}/{train_loader.Count}]");
        i++;
    }
    double epoch_loss = running_loss / train_dataset.Count;
    Console.WriteLine($"Train Loss: {epoch_loss:F4}");

    model.eval();
    long correct = 0;
    int total = 0;
    using (torch.no_grad())
    {
        foreach (var item in val_loader)
        {
            var (inputs, labels) = (item["data"], item["label"]);

            var inputs_device = inputs.to(defaultDevice);
            var labels_device = labels.to(defaultDevice);
            var outputs = model.call(inputs_device);
            var predicted = outputs.argmax(1);
            total += (int)labels.size(0);
            correct += (predicted == labels_device).sum().item<long>();
        }
    }

    double val_accuracy = 100.0 * correct / total;
    Console.WriteLine($"Validation Accuracy: {val_accuracy:F2}%");
}

model.save("model.dat");

#endregion

model.load("model.dat");
model.to(device: defaultDevice);
model.eval();


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

List<Tensor> imgs = new();
imgs.Add(transform.call(MM.LoadImage("airplane.jpg").to(defaultDevice)).view(1, 3, 32, 32));
imgs.Add(transform.call(MM.LoadImage("cat.jpg").to(defaultDevice)).view(1, 3, 32, 32));
imgs.Add(transform.call(MM.LoadImage("dog.jpg").to(defaultDevice)).view(1, 3, 32, 32));

using (torch.no_grad())
{

    foreach (var data in imgs)
    {
        var outputs = model.call(data);

        var index = outputs[0].argmax(0).ToInt32();

        // 转换为归一化的概率
        // outputs.shape = [1,10]，所以取 [dim:1]
        var array = torch.nn.functional.softmax(outputs, dim: 1);
        var max = array[0].ToFloat32Array();
        var predicted1 = classes[index];
        Console.WriteLine($"识别结果 {predicted1}，准确率：{max[index] * 100}%");
    }
}
