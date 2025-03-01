#pragma warning disable CS1591 // 缺少对公共可见类型或成员的 XML 注释

using TorchSharp.Modules;
using TorchSharp.PyBridge;

namespace Maomi.Torch;

public static partial class MM
{
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
        const string modelName = "vgg13_bn.dat";
        string repoBaseUrl = DatReposityBase;
        if (!repoBaseUrl.EndsWith('/'))
        {
            repoBaseUrl += "/";
        }

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, modelName);
        var tempFilePath = $"vgg13_bn_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, modelName);

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, repoBaseUrl + modelName, Path.Combine(torchcsharpDir, tempFilePath), modelPath);
        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as VGG)!;
    }

    public static ResNet load_vgg13bn_pth(this VGG net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        const string modelName = "alexnet-owt-7be5be79.pth";
        string repoBaseUrl = PthReposityBase;
        if (!repoBaseUrl.EndsWith('/'))
        {
            repoBaseUrl += "/";
        }

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, modelName);
        var tempFilePath = $"vgg13_bn_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, modelName).Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, repoBaseUrl + modelName, Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

}
