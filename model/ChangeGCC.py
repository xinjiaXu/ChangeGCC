from timm.layers import trunc_normal_
from model.basic import *
from model.backbone import GCC
from model.fusion import BTF


class ChangeGCC(nn.Module):
    def __init__(self, img_size=256, num_classes=1, size="base"):
        super(ChangeGCC, self).__init__()
        if size == "base":
            self.embed_dims = [40, 80, 160, 320]
            self.encoder = GCC(depths=[2, 3, 10, 3], dims=self.embed_dims, drop_path_rate=0.1, use_layer_scale=False)
        elif size == "small":
            self.embed_dims = [40, 80, 160, 320]
            self.encoder = GCC(depths=[2, 2, 7, 2], dims=self.embed_dims, drop_path_rate=0.02, use_layer_scale=True)
        elif size == "tiny":
            self.embed_dims = [32, 64, 128, 240]
            self.encoder = GCC(depths=[2, 2, 4, 2], dims=self.embed_dims, drop_path_rate=0.02, use_layer_scale=False)
        else:
            raise NotImplementedError(f"Unimplemented model size: {size}")
        self.fusion_module = BTF(img_size, self.embed_dims)
        self.decoder1 = UpBlock(self.embed_dims[3], self.embed_dims[2])
        self.decoder2 =  UpBlock(self.embed_dims[2]*2, self.embed_dims[1])
        self.decoder3 =  UpBlock(self.embed_dims[1]*2, self.embed_dims[0])
        self.decoder4 =  UpBlock(self.embed_dims[0]*2, 24)
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        # Encoder
        enc1, enc2, enc3, x = self.encoder.forward_features(torch.cat([x1, x2], dim=0))
        # Fusion
        x, enc3, enc2, enc1 = self.fusion_module([x, enc3, enc2, enc1])
        # Decoder
        dec3 = self.decoder1(x)
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        dec1 = self.decoder3(torch.cat([dec2, enc2], dim=1))
        dec0 = self.decoder4(torch.cat([dec1, enc1], dim=1))
        # Final output
        out = self.final_conv(dec0)
        return torch.sigmoid(out)


class UpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, drop=0., up=True):
        super().__init__()
        self.conv = ConvNormAct(dim_in, dim_in, kernel_size=1)
        self.conv_local = ConvNormAct(dim_in, dim_in, kernel_size=9, groups=dim_in, act_layer='silu')

        self.proj = ConvNormAct(dim_in, dim_out, kernel_size=1)
        self.proj_drop = nn.Dropout(drop)
        if up:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_local(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.upsample(x)
        return x
