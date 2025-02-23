#pragma warning disable CS1591 // 缺少对公共可见类型或成员的 XML 注释

using System.Security.Cryptography;
using TorchSharp.Modules;
using TorchSharp.PyBridge;

namespace Maomi.Torch;
public static partial class MM
{
    public static string ReposityBase { get; set; } = Huggingface;
    public static string PthReposityBase { get; set; } = HuggingfacePth;

    public const string Huggingface = "https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/";
    public const string ModelScope = "https://www.modelscope.cn/models/whuanle/torchcsharp/resolve/master/dats/";

    public const string HuggingfacePth = "https://huggingface.co/whuanle/torchcsharp/resolve/main/checkpoints/";
    public const string ModelScopePth = "https://www.modelscope.cn/models/whuanle/torchcsharp/resolve/master/checkpoints/";

    public static AlexNet LoadModel(this AlexNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadModelAsync(net, strict, skip, loadedParameters).Result;
    }

    public static AlexNet load_dat(this AlexNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadModel(net, strict, skip, loadedParameters);
    }

    public static async Task<AlexNet> LoadModelAsync(this AlexNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/alexnet.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "alexnet.dat");
        var tempFilePath = $"alexnet_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "alexnet.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as AlexNet)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "alexnet.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);

        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as AlexNet)!;
    }

    public static AlexNet load_pth(this AlexNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "alexnet-owt-7be5be79.pth");
        var tempFilePath = $"alexnet_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 =  GetModelMd5(httpClient, "alexnet-owt-7be5be79.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as AlexNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "alexnet-owt-7be5be79.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as AlexNet)!;
    }

    public static GoogleNet load_dat(this GoogleNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadModelAsync(net, strict, skip, loadedParameters).Result;
    }

    public static GoogleNet LoadModel(this GoogleNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadModelAsync(net, strict, skip, loadedParameters).Result;
    }

    public static async Task<GoogleNet> LoadModelAsync(this GoogleNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/googlenet.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "googlenet.dat");
        var tempFilePath = $"googlenet_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "googlenet.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as GoogleNet)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "googlenet.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);

        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as GoogleNet)!;
    }

    public static GoogleNet load_pth(this GoogleNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "googlenet-1378be20.pth");
        var tempFilePath = $"googlenet_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "googlenet-1378be20.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as GoogleNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "googlenet-1378be20.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as GoogleNet)!;
    }


    public static InceptionV3 load_dat(this InceptionV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadModelAsync(net, strict, skip, loadedParameters).Result;
    }

    public static InceptionV3 LoadModel(this InceptionV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadModelAsync(net, strict, skip, loadedParameters).Result;
    }

    public static async Task<InceptionV3> LoadModelAsync(this InceptionV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/inception_v3.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "inception_v3.dat");
        var tempFilePath = $"inception_v3_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "inception_v3.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as InceptionV3)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "inception_v3.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);

        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as InceptionV3)!;
    }

    public static InceptionV3 load_pth(this InceptionV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "inception_v3_google-0cc3c7bd.pth");
        var tempFilePath = $"inception_v3_google_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "inception_v3_google-0cc3c7bd.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as InceptionV3)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "inception_v3_google-0cc3c7bd.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as InceptionV3)!;
    }

    public static MobileNetV2 load_dat(this MobileNetV2 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadModelAsync(net, strict, skip, loadedParameters).Result;
    }

    public static MobileNetV2 LoadModel(this MobileNetV2 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadModelAsync(net, strict, skip, loadedParameters).Result;
    }

    public static async Task<MobileNetV2> LoadModelAsync(this MobileNetV2 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/mobilenet_v2.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "mobilenet_v2.dat");
        var tempFilePath = $"mobilenet_v2_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "mobilenet_v2.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV2)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "mobilenet_v2.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);
        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV2)!;
    }

    public static MobileNetV2 load_pth(this MobileNetV2 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "mobilenet_v2-b0353104.pth");
        var tempFilePath = $"mobilenet_v2_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "mobilenet_v2-b0353104.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV2)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "mobilenet_v2-b0353104.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV2)!;
    }

    public static MobileNetV3 load_large_dat(this MobileNetV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadMobileNetV3LargeAsync(net, strict, skip, loadedParameters).Result;
    }

    public static MobileNetV3 LoadMobileNetV3Large(this MobileNetV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadMobileNetV3LargeAsync(net, strict, skip, loadedParameters).Result;
    }

    public static async Task<MobileNetV3> LoadMobileNetV3LargeAsync(this MobileNetV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/mobilenet_v3_large.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "mobilenet_v3_large.dat");
        var tempFilePath = $"mobilenet_v3_large_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "mobilenet_v3_large.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "mobilenet_v3_large.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);
        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
    }

    public static MobileNetV3 load_large_pth(this MobileNetV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "mobilenet_v3_large-8738ca79.pth");
        var tempFilePath = $"mobilenet_v3_large_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "mobilenet_v3_large-8738ca79.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "mobilenet_v3_large-8738ca79.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
    }

    public static MobileNetV3 load_small_dat(this MobileNetV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadMobileNetV3SmallAsync(net, strict, skip, loadedParameters).Result;
    }

    public static MobileNetV3 LoadMobileNetV3Small(this MobileNetV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadMobileNetV3SmallAsync(net, strict, skip, loadedParameters).Result;
    }

    public static async Task<MobileNetV3> LoadMobileNetV3SmallAsync(this MobileNetV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/mobilenet_v3_small.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "mobilenet_v3_small.dat");
        var tempFilePath = $"mobilenet_v3_small_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "mobilenet_v3_small.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "mobilenet_v3_small.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);
        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
    }

    public static MobileNetV3 load_small_pth(this MobileNetV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "mobilenet_v3_small-047dcff4.pth");
        var tempFilePath = $"mobilenet_v3_mall_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "mobilenet_v3_small-047dcff4.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "mobilenet_v3_small-047dcff4.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
    }

    public static ResNet load_renets101_dat(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadResnet101Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static ResNet LoadResnet101(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadResnet101Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<ResNet> LoadResnet101Async(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/resnet101.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "resnet101.dat");
        var tempFilePath = $"resnet101_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "resnet101.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "resnet101.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);

        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }


    public static ResNet load_renets101_pth(this ResNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "resnet101-63fe2227.pth");
        var tempFilePath = $"resnet101_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "resnet101-63fe2227.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "resnet101-63fe2227.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static ResNet load_resnet152_pth(this ResNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "resnet152-394f9c45.pth");
        var tempFilePath = $"resnet152_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "resnet152-394f9c45.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "resnet152-394f9c45.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }


    public static ResNet load_resnet152_dat(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadResnet152Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static ResNet LoadResnet152(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadResnet152Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<ResNet> LoadResnet152Async(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/resnet152.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "resnet152.dat");
        var tempFilePath = $"resnet152_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "resnet152.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "resnet152.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);

        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static ResNet load_resnet18_pth(this ResNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "resnet18-f37072fd.pth");
        var tempFilePath = $"resnet18_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "resnet18-f37072fd.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "resnet18-f37072fd.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static ResNet load_resnet18_dat(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadResnet18Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static ResNet LoadResnet18(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadResnet18Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<ResNet> LoadResnet18Async(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/resnet18.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "resnet18.dat");
        var tempFilePath = $"resnet18_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "resnet18.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "resnet18.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static ResNet load_resnet34_pth(this ResNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "resnet34-b627a593.pth");
        var tempFilePath = $"resnet34_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "resnet34-b627a593.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "resnet34-b627a593.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static ResNet load_resnet34_dat(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadResnet34Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static ResNet LoadResnet34(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadResnet34Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<ResNet> LoadResnet34Async(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/resnet34.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "resnet34.dat");
        var tempFilePath = $"resnet34_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "resnet34.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "resnet34.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);

        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static ResNet load_resnet50_pth(this ResNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "resnet50-0676ba61.pth");
        var tempFilePath = $"resnet50_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "resnet50-0676ba61.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "resnet50-0676ba61.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static ResNet load_resnet50_dat(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadResnet50Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static ResNet LoadResnet50(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadResnet50Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<ResNet> LoadResnet50Async(this ResNet resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/resnet50.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "resnet50.dat");
        var tempFilePath = $"resnet50_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "resnet50.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "resnet50.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);

        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static ResNet load_vgg11_pth(this VGG net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg11-8a719046.pth");
        var tempFilePath = $"vgg11_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "vgg11-8a719046.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "vgg11-8a719046.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static VGG load_vgg11_dat(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG11Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static VGG LoadVGG11(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG11Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVGG11Async(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg11.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg11.dat");
        var tempFilePath = $"vgg11_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg11.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "vgg11.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);

        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }


    public static ResNet load_vgg11bn_pth(this VGG net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg11_bn-6002323d.pth");
        var tempFilePath = $"vgg11_bn_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "vgg11_bn-6002323d.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "vgg11_bn-6002323d.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }


    public static VGG load_vgg11bn_dat(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG11BNAsync(resnet, strict, skip, loadedParameters).Result;
    }

    public static VGG LoadVGG11BN(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG11BNAsync(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVGG11BNAsync(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg11_bn.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg11_bn.dat");
        var tempFilePath = $"vgg11_bn_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg11_bn.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "vgg11_bn.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }

    public static ResNet load_vgg13_pth(this VGG net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg13-19584684.pth");
        var tempFilePath = $"vgg13_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "vgg13-19584684.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "vgg13-19584684.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static VGG load_vgg13_dat(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG13Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static VGG LoadVGG13(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG13Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVGG13Async(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg13.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg13.dat");
        var tempFilePath = $"vgg13_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg13.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "vgg13.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }

    public static ResNet load_vgg13bn_pth(this VGG net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg13_bn-abd245e5.pth");
        var tempFilePath = $"vgg13_bn_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "vgg13_bn-abd245e5.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "vgg13_bn-abd245e5.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static VGG load_vgg13bn_dat(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG13BNAsync(resnet, strict, skip, loadedParameters).Result;
    }

    public static VGG LoadVGG13BN(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG13BNAsync(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVGG13BNAsync(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg13_bn.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg13_bn.dat");
        var tempFilePath = $"vgg13_bn_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg13_bn.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "vgg13_bn.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }

    public static ResNet load_vgg16_pth(this VGG net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg16-397923af.pth");
        var tempFilePath = $"vgg16_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "vgg16-397923af.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "vgg16-397923af.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static VGG load_vgg16_dat(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG16Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static VGG LoadVGG16(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG16Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVGG16Async(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg16.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg16.dat");
        var tempFilePath = $"vgg16_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg16.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "vgg16.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }

    public static ResNet load_vgg16bn_pth(this VGG net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg16_bn-6c64b313.pth");
        var tempFilePath = $"vgg16_bn_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "vgg16_bn-6c64b313.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "vgg16_bn-6c64b313.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static VGG load_vgg16bn_dat(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG16BNAsync(resnet, strict, skip, loadedParameters).Result;
    }

    public static VGG LoadVGG16BN(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG16BNAsync(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVGG16BNAsync(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg16_bn.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg16_bn.dat");
        var tempFilePath = $"vgg16_bn_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg16_bn.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "vgg16_bn.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }

    public static ResNet load_vgg19_pth(this VGG net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg19-dcbb9e9d.pth");
        var tempFilePath = $"vgg19_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "vgg19-dcbb9e9d.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "vgg19-dcbb9e9d.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static VGG load_vgg19_dat(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG19Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static VGG LoadVGG19(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG19Async(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVGG19Async(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg19.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg19.dat");
        var tempFilePath = $"vgg19_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg19.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "vgg19.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }

    public static VGG LoadVGG19BN(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG19BNAsync(resnet, strict, skip, loadedParameters).Result;
    }
    public static ResNet load_vgg19bn_pth(this VGG net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg19_bn-c79401a0.pth");
        var tempFilePath = $"vgg19_bn_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, "vgg19_bn-c79401a0.pth").Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + "vgg19_bn-c79401a0.pth", Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static VGG load_vgg19bn_dat(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        return LoadVGG19BNAsync(resnet, strict, skip, loadedParameters).Result;
    }

    public static async Task<VGG> LoadVGG19BNAsync(this VGG resnet, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        // https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/vgg19_bn.dat

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, "vgg19_bn.dat");
        var tempFilePath = $"vgg19_bn_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, "vgg19_bn.dat");

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, ReposityBase + "vgg19_bn.dat", Path.Combine(torchcsharpDir, tempFilePath), modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }

    private static string CalculateMD5(string fpath)
    {
        MD5 md5 = MD5.Create();
        byte[] hash;
        using (var stream = File.OpenRead(fpath))
        {
            hash = md5.ComputeHash(stream);
        }
        return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
    }

    private static async Task<string> GetModelMd5(HttpClient httpClient, string modelFileName)
    {
        Console.WriteLine($"Checking md5 for model file {modelFileName}");
        var md5 = await httpClient.GetStringAsync(ReposityBase + modelFileName + ".md5");
        return md5;
    }

    private static string CheckPath()
    {
        var cacheDir = Environment.GetEnvironmentVariable("CSTORCH_HOME");

        if (string.IsNullOrEmpty(cacheDir))
        {
            string userDirectory = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            cacheDir = Path.Combine(userDirectory, ".cache");
        }

        var torchcsharpDir = Path.Combine(cacheDir, "torchcsharp");
        if (!Directory.Exists(cacheDir))
        {
            Directory.CreateDirectory(cacheDir);
        }
        if (!Directory.Exists(torchcsharpDir))
        {
            Directory.CreateDirectory(torchcsharpDir);
        }

        return torchcsharpDir;
    }

    private static async Task DownloadFileWithProgressAsync(HttpClient httpClient, string url, string tempFilePath, string destinationPath, int chunkSize = 32 * 1024)
    {
        var fileName = Path.GetFileName(destinationPath);
        Console.WriteLine($"Downloading {url} to {tempFilePath}");

        using HttpResponseMessage response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        long? totalBytes = response.Content.Headers.ContentLength;
        using var contentStream = await response.Content.ReadAsStreamAsync();
        using var fileStream = File.Create(tempFilePath);

        byte[] buffer = new byte[chunkSize];
        long totalRead = 0;
        int bytesRead;
        while ((bytesRead = await contentStream.ReadAsync(buffer, 0, buffer.Length)) > 0)
        {
            await fileStream.WriteAsync(buffer, 0, bytesRead);
            totalRead += bytesRead;

            if (totalBytes.HasValue)
            {
                PrintProgress(totalRead, totalBytes.Value);
            }
        }

        fileStream.Dispose();

        File.Move(tempFilePath, destinationPath);
    }

    static void PrintProgress(long bytesRead, long totalBytes)
    {
        int totalBlocks = 50;
        double percentage = (double)bytesRead / totalBytes * 100;
        int filledBlocks = (int)(totalBlocks * bytesRead / totalBytes);

        string progressBar = "[" + new string('#', filledBlocks) + new string('-', totalBlocks - filledBlocks) + "]";
        Console.SetCursorPosition(0, Console.CursorTop);
        Console.Write($" {bytesRead}/{totalBytes} {progressBar} {percentage:F2}%");
    }
}
