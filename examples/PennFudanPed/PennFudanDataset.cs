using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torchvision;

namespace PennFudanPed;
public class PennFudanDataset : torch.utils.data.Dataset
{
    private string root;
    public string Root => root;


    private ITransform transforms;
    public ITransform Transforms => transforms;

    private List<(string, string)> images;

    public PennFudanDataset(string root,ITransform transforms)
    {
        this.root = root;
        this.transforms = transforms;

        /*
                 self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
         */
    }

    public override long Count => throw new NotImplementedException();

    public override Dictionary<string, torch.Tensor> GetTensor(long index)
    {
        throw new NotImplementedException();
    }
}
