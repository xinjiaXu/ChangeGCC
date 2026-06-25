from timm.layers import trunc_normal_
from utils.func import *
from model.basic import *


class GCC(nn.Module):
    def __init__(self, num_classes=1000, depths=None, dims=None, drop_path_rate=0.1, use_layer_scale=False):
        super().__init__()
        depths = default(depths, [2, 3, 10, 3])
        dims = default(dims, [40, 80, 160, 320])
        self.num_classes = num_classes
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(ConvBN(in_channels=3, out_channels=dims[0] // 2), nn.GELU(),
                          ConvBN(in_channels=dims[0] // 2, out_channels=dims[0], stride=1))
        )
        for i in range(len(dims) - 1):
            self.downsample_layers.append(ConvBN(in_channels=dims[i], out_channels=dims[i + 1]))
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur_dp_idx = 0
        exp_rates = [8, 8, 4, 4]
        for i in range(len(dims)):
            stage = nn.Sequential(*[
                GCCBlock(dim=dims[i], exp_rate=exp_rates[i], drop_path_rate=dp_rates[cur_dp_idx + j],
                         use_layer_scale=use_layer_scale) for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur_dp_idx += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-5)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        feature_list = []
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feature_list.append(x)
        return feature_list

    def forward(self, x):
        x = self.forward_features(x)[-1]
        x = x.mean(dim=[-2, -1])
        x = self.norm(x)
        return self.head(x)


class GCCBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, exp_rate:float=4.0, drop_path_rate=0.,
                 use_layer_scale=True, channel_mix_norm=False):
        super().__init__()
        self.token_mixer = GCConv(dim, kernel_size=kernel_size)
        self.channel_mixer = GLU(in_features=dim, hidden_features=int(2 * int(dim * exp_rate) / 3),
                                 norm=channel_mix_norm)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.layer_scale_1 = LayerScale(dim) if use_layer_scale else nn.Identity()
        self.layer_scale_2 = LayerScale(dim) if use_layer_scale else nn.Identity()
        self.norm_1 = LayerNorm2d(dim, eps=1e-6)
        self.norm_2 = LayerNorm2d(dim, eps=1e-6)

    def forward(self, x):
        x = x.contiguous()
        y = self.token_mixer(self.norm_1(x).contiguous()).contiguous()
        out = x + self.drop_path(self.layer_scale_1(y))
        out2 = self.channel_mixer(self.norm_2(out).contiguous()).contiguous()
        return out + self.drop_path(self.layer_scale_2(out2))


class GCConv(nn.Module):
    def __init__(self, dim, kernel_size=3, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.k2 = kernel_size * kernel_size
        self.padding = kernel_size // 2
        self.kernel_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kernel_act = nn.GELU()
        self.kernel_gen = nn.Linear(self.k2, self.k2, bias=bias)
        self.x_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)
        self.difference_control = nn.Parameter(torch.zeros(dim))

    def _generate_kernels(self, x):
        B, C, H, W = x.shape
        kernels = self.kernel_proj(x)
        kernels = kernels.view(B, C, H * W)
        kernels = adaptive_avg_pool1d(kernels, self.k2)
        kernels = self.kernel_act(kernels)
        kernels = self.kernel_gen(kernels)
        kernels = kernels.reshape(B, C, self.kernel_size, self.kernel_size)

        return kernels

    def _apply_kernel_difference(self, kernels):
        mean_kernels = kernels.mean(dim=(2, 3), keepdim=True)
        factor = torch.sigmoid(self.difference_control).view(1, -1, 1, 1)
        return kernels - factor * mean_kernels

    def forward(self, x):
        b, c, h, w = x.shape
        kernels = self._generate_kernels(x)
        kernels = self._apply_kernel_difference(kernels)
        x = self.x_proj(x)
        x = x.reshape(1, b * c, h, w)
        kernels = kernels.reshape(b * c, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x, kernels, padding=self.padding, groups=b * c)
        x = x.reshape(b, c, h, w)
        return self.proj(x)
