using TorchSharp.Modules;
using static TorchSharp.torch;
using nn = TorchSharp.torch.nn;

public class NeuralNetwork : nn.Module<Tensor, Tensor>
{
    // 传递给基类的参数是模型的名称
    public NeuralNetwork() : base(nameof(NeuralNetwork))
    {
        flatten = nn.Flatten();
        linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10));

        // C# 版本需要调用这个函数，将模型的组件注册到模型中
        RegisterComponents();
    }

    Flatten flatten;
    Sequential linear_relu_stack;

    public override Tensor forward(Tensor input)
    {
        // 将输入一层层处理并传递给下一层
        var x = flatten.call(input);
        var logits = linear_relu_stack.call(x);
        return logits;
    }
}