using SkiaSharp;
using System.Linq.Expressions;
using System.Reflection;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;
using static TorchSharp.torchvision.io;

namespace Maomi.Torch;

/// <summary>
/// A generic data loader where the images are arranged in this way by default.
/// </summary>
public class ImageFolderDataset : torch.utils.data.Dataset
{
    static readonly string[] extensions = new string[] { ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp" };

    torchvision.ITransform transform;
    Dictionary<string, int> class_to_idx;
    List<string> classes;
    List<(string, int)> imgs;
    string root;

    /// <summary>
    /// 
    /// </summary>
    /// <param name="root"></param>
    /// <param name="transform"></param>
    public ImageFolderDataset(string root, torchvision.ITransform transform)
    {
        this.root = root;
        this.transform = transform;

        var dirs = Directory.GetDirectories(root);

        class_to_idx = new Dictionary<string, int>();
        classes = new List<string>();
        imgs = new();

        for (int classIndex = 0; classIndex < dirs.Length; classIndex++)
        {
            var classPath = dirs[classIndex];
            var className = Path.GetFileName(classPath)!;
            classes.Add(className);
            class_to_idx.Add(className, classIndex);

            var images = Directory.GetFiles(classPath, "*.*", SearchOption.AllDirectories)
                                  .Where(s => extensions.Any(e => s.EndsWith(e)))
                                  .ToArray();

            foreach (var imagePath in images)
            {
                imgs.Add((imagePath, classIndex));
            }
        }
    }

    /// <inheritdoc/>
    public override long Count => imgs.Count;

    /// <inheritdoc/>
    public override Dictionary<string, Tensor> GetTensor(long index)
    {
        Dictionary<string, Tensor> tensors = new();
        var item = imgs[(int)index];
        Tensor? tensor = MM.LoadImage(item.Item1);

        //var lstIdx = tensor.shape.Length;
        //tensor = tensor.reshape(1, tensor.shape[lstIdx - 3], tensor.shape[lstIdx - 2], tensor.shape[lstIdx - 1]);
        if (transform is not null)
        {
            tensor = transform.call(tensor);
        }
        //tensor = tensor.squeeze(0);

        tensors["data"] = tensor;
        tensors["label"] = torch.tensor(item.Item2, ScalarType.Int64);

        return tensors;
    }
}

public static partial class MM
{
    public static class Datasets
    {
        /// <summary>
        /// A generic data loader where the images are arranged in this way by default.
        /// </summary>
        /// <param name="root"></param>
        /// <param name="target_transform"></param>
        /// <returns></returns>
        public static torch.utils.data.Dataset ImageFolder(string root, ITransform target_transform = null!)
        {
            var datasets = new ImageFolderDataset(root, target_transform);
            return datasets;
        }
    }
}

public static partial class MM
{
    /// <summary>
    /// torchvision.io.SkiaImager.ToTensor.
    /// </summary>
    /// <param name="mode"></param>
    /// <param name="bitmap"></param>
    /// <returns></returns>
    public delegate Tensor SkiaImagerToTensor(ImageReadMode mode, SKBitmap bitmap);

    /// <summary>
    /// torchvision.io.SkiaImager.ToTensor.
    /// </summary>
    public static SkiaImagerToTensor ToTensor;
    static MM()
    {
        ToTensor = BuildSkiaImagerToTensor();
    }

    private static SkiaImagerToTensor BuildSkiaImagerToTensor()
    {
        MethodInfo? toTensorMethod = typeof(torchvision.io.SkiaImager)
            .GetMethod("ToTensor", BindingFlags.NonPublic | BindingFlags.Static);

        if (toTensorMethod == null)
        {
            ArgumentNullException.ThrowIfNull(toTensorMethod, nameof(toTensorMethod));
        }

        /*
                 MethodInfo? toTensorMethod = typeof(torchvision.io.SkiaImager)
    .GetMethod("ToTensor", BindingFlags.NonPublic | BindingFlags.Static);
        var d = toTensorMethod.Invoke(null, new object[] { isTransparent ? torchvision.io.ImageReadMode.RGB : torchvision.io.ImageReadMode.RGB_ALPHA, bitmap });

         */

        ParameterExpression modeParameter = Expression.Parameter(typeof(ImageReadMode), "mode");
        ParameterExpression bitmapParameter = Expression.Parameter(typeof(SKBitmap), "bitmap");

        var arguments = new ParameterExpression[] { modeParameter, bitmapParameter };

        MethodCallExpression methodCall = Expression.Call(
            instance: null,
            method: toTensorMethod,
            arguments: arguments
        );

        Expression<SkiaImagerToTensor> lambda = Expression.Lambda<SkiaImagerToTensor>(
            methodCall,
            parameters: arguments
        );

        SkiaImagerToTensor tensorDelegate = lambda.Compile();
        return tensorDelegate;
    }
}