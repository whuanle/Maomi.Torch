using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace Maomi.Torch;

public class ReshapeTransform : ITransform
{
    private readonly long[] _shape;
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
    public static partial class transforms
    {
        public static ITransform ReshapeTransform(long[] shape)
        {
            return new ReshapeTransform(shape);
        }
    }
}