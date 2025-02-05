using SkiaSharp;
using System.Collections.Concurrent;
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
    /// <param name="channels">Number of image channels, default is 3.</param>
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
            else if (bitmap.ColorType == SKColorType.Bgra8888 || bitmap.ColorType == SKColorType.Rgba8888)
            {
                var tensorImg = ImageToTensor(bitmap, channels);
                return tensorImg;
            }

            throw new InvalidOperationException("Expected color type.");

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
    public static Tensor LoadImage(Stream stream, int channels = 3)
    {
        using (SKBitmap bitmap = SKBitmap.Decode(stream))
        {
            if (bitmap.ColorType == SKColorType.Gray8 || channels == 1)
            {
                var tensorImg = ImageToGrayTensor(bitmap);

                return tensorImg;
            }
            else if (bitmap.ColorType == SKColorType.Bgra8888 || bitmap.ColorType == SKColorType.Rgba8888)
            {
                var tensorImg = ImageToTensor(bitmap, channels);
                return tensorImg;
            }

            throw new InvalidOperationException("Expected color type.");
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
    public static async Task<Tensor> LoadImageFromUrl(string url, int channels = 3)
    {
        using HttpClient httpClient = new();
        var stream = await httpClient.GetStreamAsync(url);
        using (SKBitmap bitmap = SKBitmap.Decode(stream))
        {
            if (bitmap.ColorType == SKColorType.Gray8 || channels == 1)
            {
                var tensorImg = ImageToGrayTensor(bitmap);

                return tensorImg;
            }
            else if (bitmap.ColorType == SKColorType.Bgra8888 || bitmap.ColorType == SKColorType.Rgba8888)
            {
                var tensorImg = ImageToTensor(bitmap, channels);
                return tensorImg;
            }

            throw new InvalidOperationException("Expected color type.");
        }
    }

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
            using (SKBitmap bitmap = SKBitmap.Decode(imagePath))
            {
                Tensor? tensorImg = default;
                if (bitmap.ColorType == SKColorType.Gray8 || channels == 1)
                {
                    tensorImg = ImageToGrayTensor(bitmap);
                    tensors.Add(tensorImg);
                }
                else if (bitmap.ColorType == SKColorType.Bgra8888 || bitmap.ColorType == SKColorType.Rgba8888)
                {
                     tensorImg = ImageToTensor(bitmap, channels);
                    tensors.Add(tensorImg);
                }
                else
                {
                    throw new InvalidOperationException("Expected color type.");
                }
            }
        }

        return tensors;
    }

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
        ConcurrentDictionary<int,Tensor> tensors = new ConcurrentDictionary<int,Tensor>();
        using HttpClient httpClient = new();

        Action<int,HttpClient,string> func = async (index, httpClient, url) =>
        {
            var stream = await httpClient.GetStreamAsync(url);
            using (SKBitmap bitmap = SKBitmap.Decode(stream))
            {
                Tensor? tensorImg = default;
                if (bitmap.ColorType == SKColorType.Gray8 || channels == 1)
                {
                    tensorImg = ImageToGrayTensor(bitmap);
                }
                else if (bitmap.ColorType == SKColorType.Bgra8888 || bitmap.ColorType == SKColorType.Rgba8888)
                {
                    tensorImg = ImageToTensor(bitmap, channels);
                }
                else
                {
                    throw new InvalidOperationException("Expected color type.");
                }
                tensors[index] = tensorImg;
            }
        };

        for(int i=0;i<images.Count; i++)
        {
            func(i, httpClient, images[i]);
        }

        return tensors.OrderBy(x => x.Key).Select(x => x.Value).ToArray();
    }

    private static Tensor ImageToTensor(SKBitmap bitmap, int channels = 3)
    {
        int width = bitmap.Width;
        int height = bitmap.Height;

        float[,,] floatData = new float[channels, height, width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                SKColor color = bitmap.GetPixel(x, y);

                floatData[0, y, x] = color.Red / 255.0f; // Red channel
                floatData[1, y, x] = color.Green / 255.0f; // Green channel
                floatData[2, y, x] = color.Blue / 255.0f; // Blue channel
            }
        }

        // Convert a multidimensional array to a one-dimensional array (for tensor creation).
        // 将多维数组转换为一维数组（为了进行张量创建）.
        float[] flattenedData = new float[channels * height * width];
        Buffer.BlockCopy(floatData, 0, flattenedData, 0, flattenedData.Length * sizeof(float));

        var tensor = torch.tensor(flattenedData, new long[] { 1, channels, height, width });

        return tensor;
    }

    private static Tensor ImageToGrayTensor(SKBitmap bitmap)
    {
        int width = bitmap.Width;
        int height = bitmap.Height;

        float[,,] floatData = new float[1, height, width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                SKColor color = bitmap.GetPixel(x, y);

                // Convert to grayscale using a standard formula
                // 将彩色图像转换为灰度值（标准加权法）
                float gray = (0.299f * color.Red + 0.587f * color.Green + 0.114f * color.Blue) / 255.0f;
                floatData[0, y, x] = gray;
            }
        }

        // Convert a multidimensional array to a one-dimensional array (for tensor creation).
        // 将多维数组转换为一维数组（为了进行张量创建）.
        float[] flattenedData = new float[1 * height * width];
        Buffer.BlockCopy(floatData, 0, flattenedData, 0, flattenedData.Length * sizeof(float));

        var tensor = torch.tensor(flattenedData, new long[] { 1, height, width });

        return tensor;
    }
}
