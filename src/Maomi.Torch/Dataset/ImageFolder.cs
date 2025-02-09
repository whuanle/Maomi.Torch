using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace Maomi.Torch;

/// <summary>
/// A generic data loader where the images are arranged in this way by default.
/// </summary>
public class ImageFolderDataset : torch.utils.data.IterableDataset
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
            var className = Path.GetDirectoryName(classPath)!;
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
    public override IList<Tensor> GetTensor(long index)
    {
        List<Tensor> tensors = new List<Tensor>();
        var item = imgs[(int)index];
        Tensor? tensor = MM.LoadImage(item.Item1);

        var shape0 = tensor.shape[1];
        var shape1 = tensor.shape[2];
        var shape2 = tensor.shape[3];

        if (transform is not null)
        {
            tensors.Insert(0, transform.call(
                tensor
                .unsqueeze(0).reshape(shape0, shape1, shape2)
                .unsqueeze(0)
                ).squeeze(0));
        }
        else
        {
            tensors.Insert(0, tensor.unsqueeze(0));
        }
        tensors.Insert(1, index);

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
        /// <param name="rootPath"></param>
        /// <param name="target_transform"></param>
        /// <returns></returns>
        public static torch.utils.data.IterableDataset ImageFolder(string rootPath, ITransform target_transform = null!)
        {
            var datasets =  new ImageFolderDataset(rootPath, target_transform);
            return datasets;
        }
    }
}