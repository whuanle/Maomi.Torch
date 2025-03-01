using Maomi.Torch;
using System.Runtime.InteropServices;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torchaudio.models;
using nn = TorchSharp.torch.nn;

var defaultDevice = MM.GetOptimalDevice();

torch.set_default_device(defaultDevice);

torch.random.manual_seed(0);
torchaudio.backend.utils.set_audio_backend(new WaveAudioBackend());

// 下载模型

var bundle = torchaudio.models.wav2vec2_model(
                    extractor_mode: torchaudio.models.FeatureExtractorNormMode.group_norm,
                    extractor_conv_layer_config: new long[][] {
                        new long[] { 512, 10, 5 },

                        new long[] { 512, 3, 2 },
                        new long[] { 512, 3, 2 },
                        new long[] { 512, 3, 2 },
                        new long[] { 512, 3, 2 },

                        new long[] { 512, 2, 2 },
                        new long[] { 512, 2, 2 }
                    },
                    extractor_conv_bias: false,
                    encoder_embed_dim: 768,
                    encoder_projection_dropout: 0.1,
                    encoder_pos_conv_kernel: 128,
                    encoder_pos_conv_groups: 16,
                    encoder_num_layers: 12,
                    encoder_num_heads: 12,
                    encoder_attention_dropout: 0.1,
                    encoder_ff_interm_features: 3072,
                    encoder_ff_interm_dropout: 0.0,
                    encoder_dropout: 0.1,
                    encoder_layer_norm_first: false,
                    encoder_layer_drop: 0.05,
                    aux_num_out: 29);

bundle.to(defaultDevice);
var( waveform, sample_rate ) = torchaudio.load("steam-train-whistle-daniel_simon.wav");

var (emission, _) = bundle.call(waveform);

var decoder = new GreedyCTCDecoder(waveform);
var transcript = decoder.call(emission[0]);


public class GreedyCTCDecoder : nn.Module<Tensor, Tensor>
{
    public GreedyCTCDecoder(Tensor labels,int blank=0) : base(nameof(GreedyCTCDecoder))
    {
        this.labels = labels;
        this.blank = blank;
        RegisterComponents();
    }

    Tensor labels;
    int blank;

    public override Tensor forward(Tensor emission)
    {
        var indices = torch.argmax(emission, dim: -1);  // [num_seq,]
        (indices, _, _) = torch.unique_consecutive(indices, dim: -1);
        var filteredIndices = indices.data<long>().Where(i => i != blank).ToArray();
        var result = string.Concat(filteredIndices.Select(i => labels[i].ToString()));

        byte[] byteArray = Encoding.UTF8.GetBytes(result);
        var tensor = torch.tensor(byteArray);
        return tensor;
    }
}

public class WaveAudioBackend : torchaudio.backend.AudioBackend
{
    public override (torch.Tensor, int) load(string filepath, long frame_offset = 0, long num_frames = -1, bool normalize = true, bool channels_first = true, torchaudio.AudioFormat? format = null)
    {
        byte[] data = File.ReadAllBytes(filepath);
        // In many cases, the first 44 bytes are for RIFF header.
        short[] waveform = MemoryMarshal.Cast<byte, short>(data.AsSpan(11 * 4)).ToArray();
        return (torch.tensor(waveform).unsqueeze(0).to(torch.float32) / short.MaxValue, 16000);
    }

    public override void save(string filepath, torch.Tensor src, int sample_rate, bool channels_first = true, float? compression = null, torchaudio.AudioFormat? format = null, torchaudio.AudioEncoding? encoding = null, int? bits_per_sample = null)
    {
        throw new NotImplementedException();
    }

    public override torchaudio.AudioMetaData info(string filepath, torchaudio.AudioFormat? format = null)
    {
        throw new NotImplementedException();
    }
}