using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using static TorchSharp.torchvision.io;
using TorchSharp;

namespace Maomi.Torch;

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
    /// Reshape a tensor.
    /// </summary>
    public static partial class transforms
    {
        static transforms()
        {
            ToTensor = BuildSkiaImagerToTensor();
        }

        /// <summary>
        /// torchvision.io.SkiaImager.ToTensor.
        /// </summary>
        public static SkiaImagerToTensor ToTensor;
    }

    private static SkiaImagerToTensor BuildSkiaImagerToTensor()
    {
        MethodInfo? toTensorMethod = typeof(torchvision.io.SkiaImager)
            .GetMethod("ToTensor", BindingFlags.NonPublic | BindingFlags.Static);

        if (toTensorMethod == null)
        {
            ArgumentNullException.ThrowIfNull(toTensorMethod, nameof(toTensorMethod));
        }

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