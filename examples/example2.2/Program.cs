using TorchSharp;
using static TorchSharp.torch;
using datasets = TorchSharp.torchvision.datasets;
using transforms = TorchSharp.torchvision.transforms;
using Maomi.Torch;
using TorchSharp.Modules;

Device defaultDevice = MM.GetOptimalDevice();
torch.set_default_device(defaultDevice);

Console.WriteLine($"当前正在使用 {defaultDevice}");

// 指定训练数据集
var training_data = datasets.FashionMNIST(
    root: "data",   // 数据集在那个目录下
    train: true,    // 加载该数据集，用于训练
    download: true, // 如果数据集不存在，是否下载
    target_transform: transforms.ConvertImageDtype(ScalarType.Float32) // 指定特征和标签转换，将标签转换为Float32
    );

// 指定测试数据集
var test_data = datasets.FashionMNIST(
    root: "data",   // 数据集在那个目录下
    train: false,    // 加载该数据集，用于训练
    download: true, // 如果数据集不存在，是否下载
    target_transform: transforms.ConvertImageDtype(ScalarType.Float32) // 指定特征和标签转换，将标签转换为Float32
    );

// 分批加载图像，打乱顺序
var train_loader = torch.utils.data.DataLoader(training_data, batchSize: 64, shuffle: true, device: defaultDevice);

// 分批加载图像，不打乱顺序
var test_loader = torch.utils.data.DataLoader(test_data, batchSize: 64, shuffle: false, device: defaultDevice);

// 初始化神经网络，并使用合适的设备加载网络
var model = new NeuralNetwork();
model.to(defaultDevice);

// 定义损失函数、优化器和学习率
var loss_fn = nn.CrossEntropyLoss();
var optimizer = torch.optim.SGD(model.parameters(), learningRate: 1e-3);

// 训练的轮数
var epochs = 5;

foreach (var epoch in Enumerable.Range(0, epochs))
{
    Console.WriteLine($"Epoch {epoch + 1}\n-------------------------------");
    Train(train_loader, model, loss_fn, optimizer);
    Test(train_loader, model, loss_fn);
}

Console.WriteLine("Done!");

model.save("model.dat");
Console.WriteLine("Saved PyTorch Model State to model.dat");

model.load("model.dat");

var classes = new string[] {
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
};

model.eval();

// 加载测试数据中的第一个图片以及其标签
var x = test_data.GetTensor(0)["data"];
var y = test_data.GetTensor(0)["label"];

x.SavePng("0.png");

using (torch.no_grad())
{
    x = x.to(defaultDevice);
    var pred = model.call(x);
    var predicted = classes[pred[0].argmax(0).ToInt32()];
    var actual = classes[y.ToInt32()];
    Console.WriteLine($"Predicted: \"{predicted}\", Actual: \"{actual}\"");

}

var img = MM.LoadImage("0.png");
using (torch.no_grad())
{
    img = img.to(defaultDevice);
    var pred = model.call(img);

    // 转换为归一化的概率
    var array = torch.nn.functional.softmax(pred, dim: 0);
    var max = array.ToFloat32Array().Max();
    var predicted = classes[pred[0].argmax(0).ToInt32()];
    Console.WriteLine($"识别结果 {predicted}，概率 {max * 100}%");
}

static void Train(DataLoader dataloader, NeuralNetwork model, CrossEntropyLoss loss_fn, SGD optimizer)
{
    var size = dataloader.dataset.Count;
    model.train();

    int batch = 0;
    foreach (var item in dataloader)
    {
        var x = item["data"];
        var y = item["label"];

        // 第一步
        // 训练当前图片
        var pred = model.call(x);

        // 通过损失函数得出与真实结果的误差
        var loss = loss_fn.call(pred, y);

        // 第二步，反向传播
        loss.backward();

        // 计算梯度并优化参数
        optimizer.step();

        // 清空优化器当前的梯度
        optimizer.zero_grad();

        // 每 100 次打印损失值和当前训练的图片数量
        if (batch % 100 == 0)
        {
            loss = loss.item<float>();
            var current = (batch + 1) * x.shape[0];
            Console.WriteLine($"loss: {loss.item<float>(),7}  [{current,5}/{size,5}]");
        }

        batch++;
    }
}

static void Test(DataLoader dataloader, NeuralNetwork model, CrossEntropyLoss loss_fn)
{
    var size = (int)dataloader.dataset.Count;
    var num_batches = (int)dataloader.Count;

    // 将模型设置为评估模式
    model.eval();

    var test_loss = 0F;
    var correct = 0F;

    using (var n = torch.no_grad())
    {
        foreach (var item in dataloader)
        {
            var x = item["data"];
            var y = item["label"];

            // 使用已训练的参数预测测试数据
            var pred = model.call(x);

            // 计算损失值
            test_loss += loss_fn.call(pred, y).item<float>();
            correct += (pred.argmax(1) == y).type(ScalarType.Float32).sum().item<float>();
        }
    }

    test_loss /= num_batches;
    correct /= size;
    Console.WriteLine($"Test Error: \n Accuracy: {(100 * correct):F1}%, Avg loss: {test_loss:F8} \n");
}