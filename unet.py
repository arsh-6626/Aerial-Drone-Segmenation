import torch
import torch.nn as nn

class ConvProp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvProp, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels,out_channels,3,1,1, bias = False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace = True),
                    nn.Conv2d(out_channels, out_channels, 3,1,1, bias = False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace = True)
        )

    def forward(self,x):
        return self.conv(x)

class UNET(nn.Module):

    def __init__(self, in_channels=3 , out_channels=1 , features = [64,128,256,512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(ConvProp(in_channels, feature))
            in_channels = feature
        flip_features = features[::-1]
        for feature in flip_features:
            self.ups.append(nn.ConvTranspose2d(feature + feature, feature, kernel_size = 2, stride = 2 ))
            self.ups.append(ConvProp(feature + feature , feature))
        self.baseline = ConvProp(flip_features[0], flip_features[0]+flip_features[0])
        self.final_step = nn.Conv2d(features[0], out_channels, kernel_size=1)
    

    def forward(self, x):
        skips = []
        for downstep in self.downs:
            x = downstep(x)
            skips.append(x)
            x = self.pool(x)
        x = self.baseline(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]

            if x.shape != skip.shape:
                x =TF.resize(x, size = skip)
            cat = torch.cat((skip, x), dim = 1)
            x = self.ups[i+1](cat)

        x = self.final_step(x)
        return(self,x)

def test():
    X = torch.randn((3,1,160,160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(X)
    # print(preds.shape)
    # print(x.shape)
    print(preds)


if __name__ == "__main__":
    test()