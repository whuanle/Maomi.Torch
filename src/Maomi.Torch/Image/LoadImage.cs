using SkiaSharp;
using System.Collections.Concurrent;
using System.Reflection;
using TorchSharp;
using static TorchSharp.torch;

namespace Maomi.Torch;

/// <summary>
/// Common functions of Maomi.Torch.<br />
/// Maomi.Torch 常用功能.
/// </summary>
public static partial class MM
{
    /// <summary>
    /// Load the image and convert to Tensor.<br />
    /// 加载图片并转换为 Tensor.
    /// </summary>
    /// <param name="imagePath">Picture path.</param>
    /// <param name="channels">Number of image channels, default is 1、3、4.</param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public static Tensor load_img(string imagePath, int channels = 3) => LoadImage(imagePath, channels);

    /// <summary>
    /// Load the image and convert to Tensor.<br />
    /// 加载图片并转换为 Tensor.
    /// </summary>
    /// <param name="imagePath">Picture path.</param>
    /// <param name="channels">Number of image channels, default is 1、3、4.</param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public static Tensor LoadImage(string imagePath, int channels = 3)
    {
        using (SKBitmap bitmap = SKBitmap.Decode(imagePath))
        {
            if (bitmap.ColorType == SKColorType.Gray8 || channels == 1)
            {
                var tensorImg = ImageToGrayTensor(bitmap);

                return tensorImg;
            }

            return ImageToTensor(bitmap, channels);
        }
    }

    /// <summary>
    /// Load the image and convert to Tensor.<br />
    /// 加载图片并转换为 Tensor.
    /// </summary>
    /// <param name="stream">Picture path.</param>
    /// <param name="channels">Number of image channels, default is 3.</param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public static Tensor load_img(Stream stream, int channels = 3) => LoadImage(stream, channels);

    /// <summary>
    /// Load the image and convert to Tensor.<br />
    /// 加载图片并转换为 Tensor.
    /// </summary>
    /// <param name="stream">Picture path.</param>
    /// <param name="channels">Number of image channels, default is 3.</param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public static Tensor LoadImage(Stream stream, int channels = 3)
    {
        using (SKBitmap bitmap = SKBitmap.Decode(stream))
        {
            if (bitmap.ColorType == SKColorType.Gray8 || channels == 1)
            {
                var tensorImg = ImageToGrayTensor(bitmap);

                return tensorImg;
            }

            return ImageToTensor(bitmap, channels);
        }
    }

    /// <summary>
    /// Load the image and convert to Tensor.<br />
    /// 加载图片并转换为 Tensor.
    /// </summary>
    /// <param name="url">Picture path.</param>
    /// <param name="channels">Number of image channels, default is 3.</param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public static Task<Tensor> load_img_from_url(string url, int channels = 3) => LoadImageFromUrl(url, channels);

    /// <summary>
    /// Load the image and convert to Tensor.<br />
    /// 加载图片并转换为 Tensor.
    /// </summary>
    /// <param name="url">Picture path.</param>
    /// <param name="channels">Number of image channels, default is 3.</param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public static async Task<Tensor> LoadImageFromUrl(string url, int channels = 3)
    {
        using HttpClient httpClient = new();
        var stream = await httpClient.GetStreamAsync(url);
        return LoadImage(stream, channels);
    }

    /// <summary>
    /// Load the image and convert to Tensor.<br />
    /// 加载图片并转换为 Tensor.
    /// </summary>
    /// <param name="images">Picture path.</param>
    /// <param name="channels">Number of image channels, default is 3.</param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public static List<Tensor> load_imgs(IList<string> images, int channels = 3) => LoadImages(images, channels);

    /// <summary>
    /// Load the image and convert to Tensor.<br />
    /// 加载图片并转换为 Tensor.
    /// </summary>
    /// <param name="images">Picture path.</param>
    /// <param name="channels">Number of image channels, default is 3.</param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public static List<Tensor> LoadImages(IList<string> images, int channels = 3)
    {
        List<Tensor> tensors = new List<Tensor>();
        foreach (var imagePath in images)
        {
            Tensor? tensorImg = LoadImage(imagePath, channels);
            tensors.Add(tensorImg);
        }

        return tensors;
    }

    /// <summary>
    /// Batch load images and merge them into one Tensor.<br />
    /// 批量加载图片并合并到一个 Tensor 中.
    /// </summary>
    /// <param name="images">Picture path.</param>
    /// <param name="channels">Number of image channels, default is 3.</param>
    /// <returns></returns>
    public static Tensor compose_imgs(IList<string> images, int channels = 3) => LoadImagesCompose(images, channels);

    /// <summary>
    /// Batch load images and merge them into one Tensor.<br />
    /// 批量加载图片并合并到一个 Tensor 中.
    /// </summary>
    /// <param name="images">Picture path.</param>
    /// <param name="channels">Number of image channels, default is 3.</param>
    /// <returns></returns>
    public static Tensor LoadImagesCompose(IList<string> images, int channels = 3)
    {
        var list = LoadImages(images, channels).Select(x => x.squeeze(0));
        var batchedTensor = torch.stack(list, dim: 0);
        return batchedTensor;
    }

    /// <summary>
    /// Load the image and convert to Tensor.<br />
    /// 加载图片并转换为 Tensor.
    /// </summary>
    /// <param name="images">Picture path.</param>
    /// <param name="channels">Number of image channels, default is 3.</param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public static Tensor[] load_img_from_urls(IList<string> images, int channels = 3) => LoadImagesFromUrl(images, channels);

    /// <summary>
    /// Load the image and convert to Tensor.<br />
    /// 加载图片并转换为 Tensor.
    /// </summary>
    /// <param name="images">Picture path.</param>
    /// <param name="channels">Number of image channels, default is 3.</param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public static Tensor[] LoadImagesFromUrl(IList<string> images, int channels = 3)
    {
        ConcurrentDictionary<int, Tensor> tensors = new ConcurrentDictionary<int, Tensor>();
        using HttpClient httpClient = new();

        Action<int, HttpClient, string> func = async (index, httpClient, url) =>
        {
            using var stream = await httpClient.GetStreamAsync(url);
            Tensor? tensorImg = LoadImage(stream, channels);
            tensors[index] = tensorImg;
        };

        for (int i = 0; i < images.Count; i++)
        {
            func(i, httpClient, images[i]);
        }

        return tensors.OrderBy(x => x.Key).Select(x => x.Value).ToArray();
    }

    /// <summary>
    /// Batch load images and merge them into one Tensor.<br />
    /// 批量加载图片并合并到一个 Tensor 中.
    /// </summary>
    /// <param name="images">Picture path.</param>
    /// <param name="channels">Number of image channels, default is 3.</param>
    /// <returns></returns>
    public static Tensor load_imgs_from_urls(IList<string> images, int channels = 3) => LoadImagesFromUrlCompose(images, channels);

    /// <summary>
    /// Batch load images and merge them into one Tensor.<br />
    /// 批量加载图片并合并到一个 Tensor 中.
    /// </summary>
    /// <param name="images">Picture path.</param>
    /// <param name="channels">Number of image channels, default is 3.</param>
    /// <returns></returns>
    public static Tensor LoadImagesFromUrlCompose(IList<string> images, int channels = 3)
    {
        var list = LoadImagesFromUrl(images, channels).Select(x => x.squeeze(0));
        var batchedTensor = torch.stack(list, dim: 0);
        return batchedTensor;
    }

    private static Tensor ImageToTensor(SKBitmap bitmap, int channels = 3)
    {
        // 图像是灰色的，但以 channels = 3 方式加载
        if (bitmap.ColorType == SKColorType.Gray8)
        {
            Tensor t = torch.tensor(bitmap.Bytes, 1, bitmap.Height, bitmap.Width);
            return t.expand(3, bitmap.Height, bitmap.Width).MoveToOuterDisposeScope();
        }

        bool isTransparent = channels == 4;

        return MM.transforms.ToTensor(isTransparent ? torchvision.io.ImageReadMode.RGB_ALPHA : torchvision.io.ImageReadMode.RGB, bitmap);
    }

    private static Tensor ImageToGrayTensor(SKBitmap bitmap)
    {
        if (bitmap.ColorType == SKColorType.Gray8)
        {
            return torch.tensor(bitmap.Bytes, 1, bitmap.Height, bitmap.Width);
        }

        // RGB，但是使用 channel = 1
        return torchvision.transforms.Grayscale().call(torch.tensor(bitmap.Bytes, 1, bitmap.Height, bitmap.Width));
    }
}
