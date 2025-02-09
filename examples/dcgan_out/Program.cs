using dcgan;
using Maomi.Torch;
using TorchSharp;
using static TorchSharp.torch;

Device defaultDevice = MM.GetOpTimalDevice();
torch.set_default_device(defaultDevice);

// Set random seed for reproducibility
var manualSeed = 999;

// manualSeed = random.randint(1, 10000) # use if you want new results
Console.WriteLine("Random Seed:" + manualSeed);
random.manual_seed(manualSeed);
torch.manual_seed(manualSeed);


Options options = new Options()
{
    Dataroot = "E:\\datasets\\celeba",
    // Windows 下要设置这个，否则会报错
    Workers = 0,
    BatchSize = 128,
};

var netG = new dcgan.Generator(options);
netG.load("netG_epoch_24.pth");

// 生成随机噪声
var fixed_noise = torch.randn(64, options.Nz, 1, 1, device: defaultDevice);

// 生成图像
var fake_images = netG.call(fixed_noise);

fake_images.SavePng("fake_images.png");