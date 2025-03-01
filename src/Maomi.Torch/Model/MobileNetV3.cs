#pragma warning disable CS1591 // 缺少对公共可见类型或成员的 XML 注释

using TorchSharp.Modules;
using TorchSharp.PyBridge;

namespace Maomi.Torch;

public static partial class MM
{
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
        const string modelName = "mobilenet_v3_large.dat";
        string repoBaseUrl = DatReposityBase;
        if (!repoBaseUrl.EndsWith('/'))
        {
            repoBaseUrl += "/";
        }

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, modelName);
        var tempFilePath = $"mobilenet_v3_large_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, modelName);

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, repoBaseUrl + modelName, Path.Combine(torchcsharpDir, tempFilePath), modelPath);
        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
    }

    public static MobileNetV3 load_large_pth(this MobileNetV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        const string modelName = "mobilenet_v3_large-8738ca79.pth";
        string repoBaseUrl = PthReposityBase;
        if (!repoBaseUrl.EndsWith('/'))
        {
            repoBaseUrl += "/";
        }

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, modelName);
        var tempFilePath = $"mobilenet_v3_large_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, modelName).Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, PthReposityBase + modelName, Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

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
        const string modelName = "mobilenet_v3_small.dat";
        string repoBaseUrl = DatReposityBase;
        if (!repoBaseUrl.EndsWith('/'))
        {
            repoBaseUrl += "/";
        }

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, modelName);
        var tempFilePath = $"mobilenet_v3_small_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, modelName);

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, repoBaseUrl + modelName, Path.Combine(torchcsharpDir, tempFilePath), modelPath);
        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
    }

    public static MobileNetV3 load_small_pth(this MobileNetV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        const string modelName = "mobilenet_v3_small-047dcff4.pth";
        string repoBaseUrl = PthReposityBase;
        if (!repoBaseUrl.EndsWith('/'))
        {
            repoBaseUrl += "/";
        }

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, modelName);
        var tempFilePath = $"mobilenet_v3_mall_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, modelName).Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, repoBaseUrl + modelName, Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as MobileNetV3)!;
    }

}
