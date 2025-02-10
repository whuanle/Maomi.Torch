using TorchSharp;
using static TorchSharp.torch;

namespace Maomi.Torch.Tests;

public class ImageFolderDatasetTests
{
    static ImageFolderDatasetTests()
    {
        if (Directory.Exists("image_folder_dataset"))
        {
            Directory.Delete("image_folder_dataset", true);
        }

        Directory.CreateDirectory("image_folder_dataset");
        Directory.CreateDirectory("image_folder_dataset/class");
    }

    [Fact]
    public void TestImageFolderDatasetInitialization()
    {
        string rootPath = "image_folder_dataset/class";

        var transform = torchvision.transforms.ConvertImageDtype(ScalarType.Float32);
        List<string> files = new List<string>();
        for (int i = 0; i < 10; i++)
        {
            var path = $"{rootPath}/{i}.png";
            files.Add(path);
            var sourceImg = torch.rand(new long[] { 1, 3, 1280, 720 });
            sourceImg.SavePng(path);
        }

        var dataset = new ImageFolderDataset("image_folder_dataset", transform);

        Assert.Equal(10, dataset.Count);

        for (int i = 0; i < 10; i++)
        {
            var tensors = dataset.GetTensor(i);
            var data = tensors[0];
            Assert.Equal(new long[] { 3, 1280, 720 }, data.shape);
        }
    }
}
