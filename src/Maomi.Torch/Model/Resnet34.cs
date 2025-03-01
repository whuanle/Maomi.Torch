#pragma warning disable CS1591 // 缺少对公共可见类型或成员的 XML 注释

using TorchSharp.Modules;
using TorchSharp.PyBridge;

namespace Maomi.Torch;

public static partial class MM
{
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
        const string modelName = "resnet34.dat";
        string repoBaseUrl = DatReposityBase;
        if (!repoBaseUrl.EndsWith('/'))
        {
            repoBaseUrl += "/";
        }

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, modelName);
        var tempFilePath = $"resnet34_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, modelName);

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, repoBaseUrl + modelName, Path.Combine(torchcsharpDir, tempFilePath), modelPath);

        return (resnet.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as ResNet)!;
    }

    public static ResNet load_resnet34_pth(this ResNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        const string modelName = "resnet34-b627a593.pth";
        string repoBaseUrl = PthReposityBase;
        if (!repoBaseUrl.EndsWith('/'))
        {
            repoBaseUrl += "/";
        }

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, modelName);
        var tempFilePath = $"resnet34_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

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
