using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace dcgan;
public class Options
{
    /// <summary>
    /// Root directory for dataset
    /// </summary>
    public string Dataroot { get; set; } = "data/celeba";

    /// <summary>
    /// Number of workers for dataloader
    /// </summary>
    public int Workers { get; set; } = 2;

    /// <summary>
    /// Batch size during training
    /// </summary>
    public int BatchSize { get; set; } = 128;

    /// <summary>
    /// Spatial size of training images. All images will be resized to this size using a transformer.
    /// </summary>
    public int ImageSize { get; set; } = 64;

    /// <summary>
    /// Number of channels in the training images. For color images this is 3
    /// </summary>
    public int Nc { get; set; } = 3;

    /// <summary>
    /// Size of z latent vector (i.e. size of generator input)
    /// </summary>
    public int Nz { get; set; } = 100;

    /// <summary>
    /// Size of feature maps in generator
    /// </summary>
    public int Ngf { get; set; } = 64;

    /// <summary>
    /// Size of feature maps in discriminator
    /// </summary>
    public int Ndf { get; set; } = 64;

    /// <summary>
    /// Number of training epochs
    /// </summary>
    public int NumEpochs { get; set; } = 5;

    /// <summary>
    /// Learning rate for optimizers
    /// </summary>
    public double Lr { get; set; } = 0.0002;

    /// <summary>
    /// Beta1 hyperparameter for Adam optimizers
    /// </summary>
    public double Beta1 { get; set; } = 0.5;

    /// <summary>
    /// Number of GPUs available. Use 0 for CPU mode.
    /// </summary>
    public int Ngpu { get; set; } = 1;
}
