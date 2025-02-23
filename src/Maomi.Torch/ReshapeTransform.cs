using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace Maomi.Torch;

/// <summary>
/// Reshape transform.
/// </summary>
public class ReshapeTransform : ITransform
{
    private readonly long[] _shape;

    /// <summary>
    /// Reshape transform.
    /// </summary>
    /// <param name="shape"></param>
    public ReshapeTransform(long[] shape)
    {
        _shape = shape;
    }

    public Tensor call(Tensor input)
    {
        return input.view(_shape);
    }
}

/// <summary>
/// extensions.
/// </summary>
public static partial class MM
{
    /// <summary>
    /// Reshape a tensor.
    /// </summary>
    public static partial class transforms
    {

        /// <summary>
        /// Reshape a tensor.
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static ITransform ReshapeTransform(long[] shape)
        {
            return new ReshapeTransform(shape);
        }
    }
}