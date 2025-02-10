using TorchSharp;

namespace Maomi.Torch.Tests;

public class SaveImageTests
{
    static SaveImageTests()
    {
        if (Directory.Exists("save_image"))
        {
            Directory.Delete("save_image", true);
        }

        Directory.CreateDirectory("save_image");
    }

    [Fact]
    public void SavePng_ShouldCreateFile()
    {
        string imagePath = "save_image/test_image.png";
        var sourceImg = torch.rand(new long[] { 1, 3, 1280, 720 });
        sourceImg.SavePng(imagePath);

        Assert.True(File.Exists(imagePath));
    }

    [Fact]
    public void SaveJpeg_ShouldCreateFile()
    {
        string imagePath = "save_image/test_image.png";
        var sourceImg = torch.rand(new long[] { 1, 3, 1280, 720 });
        sourceImg.SaveJpeg(imagePath);

        Assert.True(File.Exists(imagePath));
    }
}
