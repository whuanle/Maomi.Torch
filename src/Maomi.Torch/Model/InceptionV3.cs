#pragma warning disable CS1591 // 缺少对公共可见类型或成员的 XML 注释

using TorchSharp.Modules;
using TorchSharp.PyBridge;

namespace Maomi.Torch;

public static partial class MM
{
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
        const string modelName = "inception_v3.dat";
        string repoBaseUrl = DatReposityBase;
        if (!repoBaseUrl.EndsWith('/'))
        {
            repoBaseUrl += "/";
        }

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, modelName);
        var tempFilePath = $"inception_v3_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, modelName);

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as InceptionV3)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, repoBaseUrl + modelName, Path.Combine(torchcsharpDir, tempFilePath), modelPath);

        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as InceptionV3)!;
    }

    public static InceptionV3 load_pth(this InceptionV3 net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        const string modelName = "inception_v3_google-0cc3c7bd.pth";
        string repoBaseUrl = PthReposityBase;
        if (!repoBaseUrl.EndsWith('/'))
        {
            repoBaseUrl += "/";
        }

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, modelName);
        var tempFilePath = $"inception_v3_google_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, modelName).Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as InceptionV3)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, repoBaseUrl + modelName, Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as InceptionV3)!;
    }

}
