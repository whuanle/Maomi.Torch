using TorchSharp.Modules;
using static TorchSharp.torch;

namespace dcgan;

public class Generator : nn.Module<Tensor, Tensor>, IDisposable
{
    private readonly Options _options;

    public Generator(Options options) : base(nameof(Generator))
    {
        _options = options;
        main = nn.Sequential(
            // input is Z, going into a convolution
            nn.ConvTranspose2d(options.Nz, options.Ngf * 8, 4, 1, 0, bias: false),
            nn.BatchNorm2d(options.Ngf * 8),
            nn.ReLU(true),
            // state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(options.Ngf * 8, options.Ngf * 4, 4, 2, 1, bias: false),
            nn.BatchNorm2d(options.Ngf * 4),
            nn.ReLU(true),
            // state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(options.Ngf * 4, options.Ngf * 2, 4, 2, 1, bias: false),
            nn.BatchNorm2d(options.Ngf * 2),
            nn.ReLU(true),
            // state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(options.Ngf * 2, options.Ngf, 4, 2, 1, bias: false),
            nn.BatchNorm2d(options.Ngf),
            nn.ReLU(true),
            // state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(options.Ngf, options.Nc, 4, 2, 1, bias: false),
            nn.Tanh()
            // state size. (nc) x 64 x 64
            );

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return main.call(input);
    }

    Sequential main;
}
