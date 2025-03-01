#pragma warning disable CS1591 // 缺少对公共可见类型或成员的 XML 注释

using TorchSharp.Modules;
using TorchSharp.PyBridge;

namespace Maomi.Torch;

public static partial class MM
{
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
        const string modelName = "googlenet.dat";
        string repoBaseUrl = DatReposityBase;
        if (!repoBaseUrl.EndsWith('/'))
        {
            repoBaseUrl += "/";
        }

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, modelName);
        var tempFilePath = $"googlenet_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.dat";

        var modelMd5 = await GetModelMd5(httpClient, modelName);

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as GoogleNet)!;
            }

            File.Delete(modelPath);
        }

        await DownloadFileWithProgressAsync(httpClient, repoBaseUrl + modelName, Path.Combine(torchcsharpDir, tempFilePath), modelPath);

        return (net.load(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as GoogleNet)!;
    }

    public static GoogleNet load_pth(this GoogleNet net, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null)
    {
        const string modelName = "googlenet-1378be20.pth";
        string repoBaseUrl = PthReposityBase;
        if (!repoBaseUrl.EndsWith('/'))
        {
            repoBaseUrl += "/";
        }

        var torchcsharpDir = CheckPath();

        HttpClient httpClient = new HttpClient();
        var modelPath = Path.Combine(torchcsharpDir, modelName);
        var tempFilePath = $"googlenet_{DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString("x16")}.pth";

        var modelMd5 = GetModelMd5(httpClient, modelName).Result;

        if (File.Exists(modelPath))
        {
            var localFileMd5 = CalculateMD5(modelPath);

            if (string.Equals(modelMd5, localFileMd5, StringComparison.OrdinalIgnoreCase))
            {
                return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as GoogleNet)!;
            }

            File.Delete(modelPath);
        }

        DownloadFileWithProgressAsync(httpClient, modelName + modelName, Path.Combine(torchcsharpDir, tempFilePath), modelPath).Wait();

        return (net.load_py(location: modelPath, strict: strict, skip: skip, loadedParameters: loadedParameters) as GoogleNet)!;
    }

}
