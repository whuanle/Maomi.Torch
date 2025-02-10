using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;
using static TorchSharp.torchvision.io;

namespace Maomi.Torch;

public static partial class MM
{
    private static readonly SkiaImager DefaultImager = new();

    /// <summary>
    /// Save the tensor data as a .png image file。<br />
    /// 将张量数据保存为 .png 图片文件.
    /// </summary>
    /// <param name="imageTensor"></param>
    /// <param name="filePath"></param>
    public static void SavePng(this Tensor imageTensor, string filePath)
    {
        SaveImage(imageTensor, filePath, torchvision.ImageFormat.Png);
    }

    /// <summary>
    /// Save the tensor data as a .jpeg image file。<br />
    /// 将张量数据保存为 .jpeg 图片文件.
    /// </summary>
    /// <param name="imageTensor"></param>
    /// <param name="filePath"></param>
    public static void SaveJpeg(this Tensor imageTensor, string filePath)
    {
        SaveImage(imageTensor, filePath, torchvision.ImageFormat.Jpeg);
    }

    /// <summary>
    /// 将张量数据保存为图片文件.
    /// </summary>
    /// <param name="imageTensor"></param>
    /// <param name="filePath">图片路径.</param>
    /// <param name="imageFormat">图像格式.</param>
    /// <param name="quality">图像质量.</param>
    public static void SaveImage(this Tensor imageTensor, string filePath, torchvision.ImageFormat imageFormat, int quality = 75)
    {
        var shapeSize = imageTensor.shape;

        // N Batch size, number of C channels, H height, and W width.
        // N 批大小、C 通道数、H 高度、W 宽度.
        var (N, C, H, W) = (0L, 0L, 0L, 0L);

        if (shapeSize.Length == 3)
        {
            (C, H, W) = (shapeSize[0], shapeSize[1], shapeSize[2]);
        }
        else if (shapeSize.Length == 4)
        {
            (N, C, H, W) = (shapeSize[0], shapeSize[1], shapeSize[2], shapeSize[3]);
        }
        else
        {
            // 张量数据维度不正确，应为 3 或 4 维.
            throw new ArgumentException("The tensor data dimension is incorrect and should be 3 or 4 dimensional.");
        }

        save_image(imageTensor, filePath, imageFormat, imager: new SkiaImager(quality));
    }

    public static void save_image(torch.Tensor tensor, string filename, ImageFormat format, long nrow = 8L, int padding = 2, bool normalize = false, (double low, double high)? value_range = null, bool scale_each = false, double pad_value = 0.0, io.Imager imager = null)
    {
        using FileStream filestream = new FileStream(filename, FileMode.OpenOrCreate);
        save_image(tensor, filestream, format, nrow, padding, normalize, value_range, scale_each, pad_value, imager);
    }

    public static void save_image(torch.Tensor tensor, Stream filestream, ImageFormat format, long nrow = 8L, int padding = 2, bool normalize = false, (double low, double high)? value_range = null, bool scale_each = false, double pad_value = 0.0, io.Imager imager = null)
    {
        using (torch.NewDisposeScope())
        {
            torch.Tensor image = torchvision.utils.make_grid(tensor, nrow, padding, normalize, value_range, scale_each, pad_value).mul(255).add_(0.5).clamp_(0, 255)
                .to(torch.uint8, torch.CPU);
            (imager ?? new SkiaImager()).EncodeImage(image, format, filestream);
        }
    }
}
