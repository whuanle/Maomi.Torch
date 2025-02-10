using Microsoft.AspNetCore.Routing;
using TorchSharp;

namespace Maomi.Torch.Tests;

public class LoadImageTests
{
    static LoadImageTests()
    {
        if (Directory.Exists("load_image"))
        {
            Directory.Delete("load_image", true);
        }

        Directory.CreateDirectory("load_image");
    }

    [Fact]
    public void LoadImage_FromFilePath_ReturnsTensor()
    {
        string imagePath = "load_image/test_image.png";

        var sourceImg = torch.rand(new long[] { 1, 3, 1280, 720 });

        sourceImg.SavePng(imagePath);
        var tensor = MM.LoadImage(imagePath);
        Assert.NotNull(tensor);
        Assert.Equal(new long[] { 3, 1280, 720 }, tensor.shape);
    }

    [Fact]
    public void LoadImage_FromStream_ReturnsTensor()
    {
        string imagePath = "load_image/test_stream_image.png";
        var sourceImg = torch.rand(new long[] { 1, 3, 1280, 720 });
        sourceImg.SavePng(imagePath);

        using var stream = File.OpenRead(imagePath);
        var tensor = MM.LoadImage(stream);
        Assert.NotNull(tensor);
        Assert.Equal(new long[] { 3, 1280, 720 }, tensor.shape);
    }

    [Fact]
    public void LoadImage_FromFilePath_Channels1_ReturnsTensor()
    {
        string imagePath = "load_image/test_image_channels1.png";

        var sourceImg = torch.rand(new long[] { 1, 3, 1280, 720 });

        sourceImg.SavePng(imagePath);
        var tensor = MM.LoadImage(imagePath, channels: 1);
        Assert.NotNull(tensor);
        Assert.Equal(new long[] { 1, 1280, 720 }, tensor.shape);
    }

    [Fact]
    public void LoadImage_FromStream_Channels1_ReturnsTensor()
    {
        string imagePath = "load_image/test_image_channels1.png";
        var sourceImg = torch.rand(new long[] { 1, 3, 1280, 720 });
        sourceImg.SavePng(imagePath);

        using var stream = File.OpenRead(imagePath);
        var tensor = MM.LoadImage(stream, channels: 1);
        Assert.NotNull(tensor);
        Assert.Equal(new long[] { 1, 1280, 720 }, tensor.shape);
    }

    [Fact]
    public void LoadImage_FromFilePath_Channels3_ReturnsTensor()
    {
        string imagePath = "load_image/test_image_channels3.png";

        var sourceImg = torch.rand(new long[] { 1, 3, 1280, 720 });

        sourceImg.SavePng(imagePath);
        var tensor = MM.LoadImage(imagePath, channels: 3);
        Assert.NotNull(tensor);
        Assert.Equal(new long[] { 3, 1280, 720 }, tensor.shape);
    }

    [Fact]
    public void LoadImage_FromStream_Channels3_ReturnsTensor()
    {
        string imagePath = "load_image/test_image_channels3.png";
        var sourceImg = torch.rand(new long[] { 1, 3, 1280, 720 });
        sourceImg.SavePng(imagePath);

        using var stream = File.OpenRead(imagePath);
        var tensor = MM.LoadImage(stream, channels: 3);
        Assert.NotNull(tensor);
        Assert.Equal(new long[] { 3, 1280, 720 }, tensor.shape);
    }

    [Fact]
    public void LoadImages_FromFiles_ReturnsTensors()
    {
        string imagePath1 = "load_image/test_images_1.png";
        var sourceImg1 = torch.rand(new long[] { 1, 3, 1280, 720 });
        sourceImg1.SavePng(imagePath1);

        string imagePath2 = "load_image/test_images_2.png";
        var sourceImg2 = torch.rand(new long[] { 1, 3, 1280, 720 });
        sourceImg2.SavePng(imagePath2);

        IList<string> images = new List<string> { imagePath1, imagePath2 };
        var tensors = MM.LoadImages(images);
        Assert.Equal(2, tensors.Count);
        foreach (var tensor in tensors)
        {
            Assert.NotNull(tensor);
            Assert.Equal(new long[] { 3, 1280, 720 }, tensor.shape);
        }
    }

    [Fact]
    public void LoadImagesCompose_FromFiles_ReturnsBatchedTensor()
    {
        string imagePath1 = "load_image/test_images_1.png";
        var sourceImg1 = torch.rand(new long[] { 1, 3, 1280, 720 });
        sourceImg1.SavePng(imagePath1);

        string imagePath2 = "load_image/test_images_2.png";
        var sourceImg2 = torch.rand(new long[] { 1, 3, 1280, 720 });
        sourceImg2.SavePng(imagePath2);

        IList<string> images = new List<string> { imagePath1, imagePath2 };

        var batchedTensor = MM.LoadImagesCompose(images);
        Assert.NotNull(batchedTensor);
        Assert.Equal(new long[] { 2, 3, 1280, 720 }, batchedTensor.shape);
    }
}
