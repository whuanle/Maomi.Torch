using dcgan;
using Maomi.Torch;
using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

// 使用 GPU 启动
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

if(Directory.Exists("samples"))
{
    Directory.Delete("samples", true);
}

Directory.CreateDirectory("samples");

var dataset = MM.Datasets.ImageFolder(options.Dataroot, torchvision.transforms.Compose(
    torchvision.transforms.Resize(options.ImageSize),
    torchvision.transforms.CenterCrop(options.ImageSize),
    torchvision.transforms.ConvertImageDtype(ScalarType.Float32),
    torchvision.transforms.Normalize(new double[] { 0.5, 0.5, 0.5 }, new double[] { 0.5, 0.5, 0.5 }))
);

var dataloader = torch.utils.data.DataLoader(dataset, batchSize: options.BatchSize, shuffle: true, num_worker: options.Workers, device: defaultDevice);

var netG = new dcgan.Generator(options).to(defaultDevice);
netG.apply(weights_init);
Console.WriteLine(netG);


var netD = new dcgan.Discriminator(options).to(defaultDevice);
netD.apply(weights_init);
Console.WriteLine(netD);

var criterion = nn.BCELoss();
var fixed_noise = torch.randn(new long[] { options.BatchSize, options.Nz, 1, 1 }, device: defaultDevice);
var real_label = 1.0;
var fake_label = 0.0;
var optimizerD = torch.optim.Adam(netD.parameters(), lr: options.Lr, beta1: options.Beta1, beta2: 0.999);
var optimizerG = torch.optim.Adam(netG.parameters(), lr: options.Lr, beta1: options.Beta1, beta2: 0.999);

var img_list = new List<Tensor>();
var G_losses = new List<double>();
var D_losses = new List<double>();

Console.WriteLine("Starting Training Loop...");

Stopwatch stopwatch = new();
stopwatch.Start();
int i = 0;
// For each epoch
for (int epoch = 0; epoch < options.NumEpochs; epoch++)
{
    // For each batch in the dataloader
    foreach (var item in dataloader)
    {
        var data = item[0];

        netD.zero_grad();
        // Format batch
        var real_cpu = data.to(defaultDevice);
        var b_size = real_cpu.size(0);
        var label = torch.full(new long[] { b_size }, real_label, dtype: ScalarType.Float32, device: defaultDevice);
        // Forward pass real batch through D
        var output = netD.forward(real_cpu);
        // Calculate loss on all-real batch
        var errD_real = criterion.call(output, label);
        // Calculate gradients for D in backward pass
        errD_real.backward();
        var D_x = output.mean().item<float>();

        // Train with all-fake batch
        // Generate batch of latent vectors
        var noise = torch.randn(new long[] { b_size, options.Nz, 1, 1 }, device: defaultDevice);
        // Generate fake image batch with G
        var fake = netG.call(noise);
        label.fill_(fake_label);
        // Classify all fake batch with D
        output = netD.call(fake.detach());
        // Calculate D's loss on the all-fake batch
        var errD_fake = criterion.call(output, label);
        // Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward();
        var D_G_z1 = output.mean().item<float>();
        // Compute error of D as sum over the fake and the real batches
        var errD = errD_real + errD_fake;
        // Update D
        optimizerD.step();

        ////////////////////////////
        // (2) Update G network: maximize log(D(G(z)))
        ////////////////////////////
        netG.zero_grad();
        label.fill_(real_label);  // fake labels are real for generator cost
        // Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD.call(fake);
        // Calculate G's loss based on this output
        var errG = criterion.call(output, label);
        // Calculate gradients for G
        errG.backward();
        var D_G_z2 = output.mean().item<float>();
        // Update G
        optimizerG.step();

        // ex: [0/25][4/3166] Loss_D: 0.5676 Loss_G: 7.5972 D(x): 0.9131 D(G(z)): 0.3024 / 0.0007
        Console.WriteLine($"[{epoch}/{options.NumEpochs}][{i%dataloader.Count}/{dataloader.Count}] Loss_D: {errD.item<float>():F4} Loss_G: {errG.item<float>():F4} D(x): {D_x:F4} D(G(z)): {D_G_z1:F4} / {D_G_z2:F4}");

        // 每处理 100 批，输出一次图片效果
        if (i % 100 == 0)
        {
            real_cpu.SavePng("samples/real_samples.png");
            fake = netG.call(fixed_noise);
            real_cpu.SavePng($"samples/fake_samples_epoch_{epoch:D3}.png");
        }

        i++;
    }


    netG.save($"samples/netg_{epoch}.dat");
    netD.save($"samples/netd_{epoch}.dat");
}

Console.WriteLine("Training finished.");
stopwatch.Stop();
Console.WriteLine($"Training Time: {stopwatch.Elapsed}");


netG.save("samples/netg.dat");
netD.save("samples/netd.dat");

static void weights_init(nn.Module m)
{
    var classname = m.GetType().Name;
    if (classname.Contains("Conv"))
    {
        if (m is Conv2d conv2d)
        {
            nn.init.normal_(conv2d.weight, 0.0, 0.02);
        }
    }
    else if (classname.Contains("BatchNorm"))
    {
        if (m is BatchNorm2d batchNorm2d)
        {
            nn.init.normal_(batchNorm2d.weight, 1.0, 0.02);
            nn.init.zeros_(batchNorm2d.bias);
        }
    }
}
