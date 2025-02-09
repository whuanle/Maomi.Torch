using TorchSharp.Modules;
using static TorchSharp.torch;

namespace dcgan;

public class Discriminator : nn.Module<Tensor, Tensor>, IDisposable
{
    private readonly Options _options;

    public Discriminator(Options options) : base(nameof(Discriminator))
    {
        _options = options;

        main = nn.Sequential(
            // input is (nc) x 64 x 64
            nn.Conv2d(options.Nc, options.Ndf, 4, 2, 1, bias: false),
            nn.LeakyReLU(0.2, inplace: true),
            // state size. (ndf) x 32 x 32
            nn.Conv2d(options.Ndf, options.Ndf * 2, 4, 2, 1, bias: false),
            nn.BatchNorm2d(options.Ndf * 2),
            nn.LeakyReLU(0.2, inplace: true),
            // state size. (ndf*2) x 16 x 16
            nn.Conv2d(options.Ndf * 2, options.Ndf * 4, 4, 2, 1, bias: false),
            nn.BatchNorm2d(options.Ndf * 4),
            nn.LeakyReLU(0.2, inplace: true),
            // state size. (ndf*4) x 8 x 8
            nn.Conv2d(options.Ndf * 4, options.Ndf * 8, 4, 2, 1, bias: false),
            nn.BatchNorm2d(options.Ndf * 8),
            nn.LeakyReLU(0.2, inplace: true),
            // state size. (ndf*8) x 4 x 4
            nn.Conv2d(options.Ndf * 8, 1, 4, 1, 0, bias: false),
            nn.Sigmoid()
            );

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var output =  main.call(input);

        return output.view(-1, 1).squeeze(1);
    }

    Sequential main;
}