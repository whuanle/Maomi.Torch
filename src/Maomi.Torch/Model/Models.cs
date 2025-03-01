#pragma warning disable CS1591 // 缺少对公共可见类型或成员的 XML 注释

using System.Security.Cryptography;
using TorchSharp.Modules;
using TorchSharp.PyBridge;

namespace Maomi.Torch;
public static partial class MM
{
    /// <summary>
    /// Set the.dat file download location<br />设置 .dat 文件下载位置.
    /// </summary>
    public static string DatReposityBase { get; set; } = HuggingfaceDat;

    /// <summary>
    /// Set the .pth file download location<br />设置 .pth 文件下载位置.
    /// </summary>
    public static string PthReposityBase { get; set; } = HuggingfacePth;

    public const string HuggingfaceDat = "https://huggingface.co/whuanle/torchcsharp/resolve/main/dats/";
    public const string ModelScopeDat = "https://www.modelscope.cn/models/whuanle/torchcsharp/resolve/master/dats/";

    public const string HuggingfacePth = "https://huggingface.co/whuanle/torchcsharp/resolve/main/checkpoints/";
    public const string ModelScopePth = "https://www.modelscope.cn/models/whuanle/torchcsharp/resolve/master/checkpoints/";


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
        var md5 = await httpClient.GetStringAsync(DatReposityBase + modelFileName + ".md5");
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
