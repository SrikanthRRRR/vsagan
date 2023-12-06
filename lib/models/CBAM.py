import torch.nn as nn
import torch
import torch.nn.functional as F

class CBAM(nn.Module):
    def forward(self, f):
        channel_att = self.channel_att(f)
        ch_at_f = channel_att * f
        spatial_att = self.spatial_att(ch_at_f)
        sp_at_f = spatial_att * ch_at_f
        return sp_at_f

    def __init__(self, n_in_channel, reduction_ratio, kernel_size):
        super(CBAM, self).__init__()
        self.n_in_channel = n_in_channel
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_att = ChannelAttention(self.n_in_channel, self.reduction_ratio)
        self.spatial_att = SpatialAttention(self.kernel_size)




class SpatialAttention(nn.Module):
    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim = 1)))

    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = kernel_size, padding= int((kernel_size-1)/2))
        # batchnorm

class ChannelAttention(nn.Module):
    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel )
        max_pool = F.max_pool2d(x, kernel)
        
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)
        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1,1,kernel[0], kernel[1])
        return out


    def __init__(self, n_in_channel, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_in_channel = n_in_channel
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_in_channel/ float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_in_channel, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_in_channel)
        )