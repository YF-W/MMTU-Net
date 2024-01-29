"""
将双卷积全部换成深度可分离卷积
"""
#!/usr/bin/env python
#coding=utf-8
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
from torchvision import models as resnet_model
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class DoubleConv(nn.Module):
    def __init__(self, in_channels,hidden_channels,out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            DepthWiseConv(in_channels,hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            DepthWiseConv(hidden_channels,out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)


class Res1(nn.Module):
    def __init__(self,in_channel,hidden_channel,out_channel):
        super(Res1,self).__init__()
        self.conv=DoubleConv(in_channel,hidden_channel,out_channel)
        self.conv1=nn.Conv2d(in_channel,out_channel,1)
    def forward(self,x):
        p=self.conv(x)
        x=self.conv1(x)
        x=x+p
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class get_qkv(nn.Module):
    def __init__(self,dim,in_channel,out_channel,heads=8,dim_head=64):
        super(get_qkv,self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

    def forward(self,x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k ,v= map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        return q,k,v


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel,kernel_size=3):
        # 这一行千万不要忘记
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)


    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim,in_channel,out_channel, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.get_qkv=get_qkv(dim,in_channel,out_channel)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        q,k,v=self.get_qkv(x)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)   #softmax
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim,in_channel,out_channel, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attention=Attention(dim,in_channel,out_channel,heads=heads, dim_head=dim_head, dropout=dropout)
        self.feed=FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):  # x为黏在一起传入transformer中的，y为传入encoder中的
        x_vit=self.norm(x)
        x_vit=self.attention(x_vit)
        x=x+x_vit
        x_vit=self.norm(x)
        x_vit=self.feed(x_vit)
        x=x+x_vit

        return x



class turn_one(nn.Module):
    def __init__(self,channels,image_size,patch_size, dim,emb_dropout=0.):
        super(turn_one,self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self,x):
        x=rearrange(x,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1=28, p2=28)
        _,_,c=x.shape
        x=nn.Linear(c, 196)(x)  # 将二维图像转化为一维的patch embeddings
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x_vit = torch.cat((cls_tokens, x), dim=1)
        x_vit += nn.Parameter(torch.randn(1, n + 1, 196))
        x_vit = self.dropout(x_vit)

        return x_vit

class vit_transformer(nn.Module):
    def __init__(self, *, in_channel,out_channel,image_size, patch_size, dim, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):   # *表示必须以关键字的形式传入参数
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches = (image_height // patch_height) * (image_width // patch_width)   # patch的数量
        # patch_dim = channels * patch_height * patch_width       # patch的维度
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
        #     nn.Linear(patch_dim, dim),
        # )
        #
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout = nn.Dropout(emb_dropout)
        self.turn1=turn_one(channels=3,image_size=image_size,patch_size=patch_size,dim=dim)
        self.turn2=turn_one(channels=64,image_size=image_size,patch_size=patch_size,dim=dim)
        # self.turn3=turn_one(channels=in_channel,image_size=image_size,patch_size=patch_size,dim=dim)
        # self.turn4=turn_one(channels=3,image_size=image_size,patch_size=patch_size,dim=dim)
        self.transformer = Transformer(dim,64,128, heads, dim_head, mlp_dim, dropout)
        self.vitLayer_UpConv = nn.ConvTranspose2d(15, 15, kernel_size=2, stride=2)
        self.vitLayer_UpConv1 = nn.ConvTranspose2d(6, 6, kernel_size=2, stride=2)
        # self.local_branch=Local(in_channel,out_channel)
        self.pool=nn.MaxPool2d(2)
    def forward(self, level1,level2,level3):
        # level1是原图，level2是传入encoder的图，level3是encoder第一层的结果
        x=level2
        for i in range(2):
            level1=self.pool(level1)   # (6,64,56,56)
        level2=self.pool(level2)  #(6,128,56,56)
        level3=self.pool(level3)  #(6,128,56,56)

        #todo将每一张图片处理成一维的
        level1=self.turn1(level1)
        level2=self.turn2(level2)
        level3=self.turn2(level3)
        x_vit=torch.cat([level1,level2,level3],dim=1)

        # 三层的结果粘在一起求q,k,input用于求v

        vit_layerInfo = []

        for i in range(4):  # 设置深度的地方[6, 64+1, dim=196]
            x_vit = self.transformer(x_vit)
            vit_layerInfo.append(x_vit)

        # x_cnn=self.local_branch(x)
        # b,n,_=vit_layerInfo[3].shape
        # v = vit_layerInfo[3].view(b, n, 14, 14)
        #
        # v = self.vitLayer_UpConv1(v)
        # x = torch.cat([v, x_cnn], dim=1)
        #
        # x = self.dw_conv1(x)
        # x = self.bn(x)
        # x = self.conv(x)
        return vit_layerInfo


class encoder2(nn.Module):
    def __init__(self,in_channel):
        super(encoder2,self).__init__()
        # self.re1=Res1(in_channel,64,64)
        # self.re2=Res1(64,128,128)
        # self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.transformer1 = vit_transformer(in_channel=128,out_channel=256,image_size=56,patch_size=28, dim=196, heads=8, mlp_dim=2048,channels=128)
        # self.transformer2 = vit_transformer(in_channel=256,out_channel=512,image_size=28, patch_size=28, dim=196, heads=8, mlp_dim=2048,channels=256)
        self.bottleneck=Bottleneck(15,256,1024)
    def forward(self,x,y,z):
        vit_layerInfo=self.transformer1(x,y,z)
        b,c,_=vit_layerInfo[3].shape
        v=vit_layerInfo[3].view(b,c,14,14)
        e=self.bottleneck(v)
        return vit_layerInfo,e


class sub_channel_attention(nn.Module):
    def __init__(self,channels,kernel_size,ratio=7):
        super(sub_channel_attention,self).__init__()
        channels2=15
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels2, out_features=channels2 // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels2 // ratio, out_features=channels2, bias=False),
        )
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels//2, kernel_size=1, stride=1, padding=0)
        self.conv1 = DepthWiseConv(channels//2*3+15,channels//2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn=nn.BatchNorm2d(channels//2)
        self.depth_conv = nn.Conv2d(in_channels=channels,
                                    out_channels=channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=1,
                                    groups=channels)
        self.decoder_unsample1=nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.decoder_unsample2=nn.ConvTranspose2d(channels//2, channels//2, kernel_size=2, stride=2)

    def forward(self,y,x,n):
        """y表示transformer返回的结果，x表示decoder返回的结果
        y(6,512,28,28)  x(6,1024,14,14)"""
        b,c,_=y.shape
        y=y.view(b,c,14,14)
        y1 = self.gap(y)
        y1 = y1.view(b,c)
        y1 = self.fc_layers(y1)
        y1=y1.view(b,c,1,1)
        y1 = self.sigmoid(y1)
        y1=y*y1.expand_as(y)
          # 得到d返回的结果
        x1=self.depth_conv(x)
        b, c, _, _ = x1.shape
        x2 = x1.detach().cpu().numpy()
        for i in range(b):
            for j in range(c):
                x2[i][j] = np.cov(x2[i][j])
        x2 = torch.tensor(x2)
        x2=x2.cuda(0)
        x2=self.conv(x2)
        x2=self.bn(x2)
        x2=self.sigmoid(x2)
        for i in range(n):
            y1=nn.Upsample(scale_factor=2,mode='bilinear')(y1)
        x=torch.cat([y1,x2,x],dim=1)
        x=self.conv1(x)
        x=self.bn(x)
        x=self.relu(x)

        return x


class Bottleneck(nn.Module):
    def __init__(self,in_channel,hidden_channel,out_channel):
        super(Bottleneck,self).__init__()
        self.bottleneck=DoubleConv(in_channel,hidden_channel,out_channel)

    def forward(self,x):
        x=self.bottleneck(x)
        return x


class encoder1(nn.Module):
    def __init__(self):
        super(encoder1,self).__init__()
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.bottleneck=Bottleneck(in_channel=512,hidden_channel=1024,out_channel=1024)


    def forward(self,x):
        skip_connection=[]
        x=self.firstconv(x)
        x=self.firstbn(x)
        x=self.firstrelu(x)
        x1=x
        e1=self.encoder1(x)
        skip_connection.append(e1)
        e2=self.encoder2(e1)
        skip_connection.append(e2)
        e3=self.encoder3(e2)
        skip_connection.append(e3)
        e4=self.encoder4(e3)
        skip_connection.append(e4)
        x=self.bottleneck(e4)

        return skip_connection,x,x1


class v1(nn.Module):
    def __init__(self,*,in_channel=3,out_channel=1):
        super(v1,self).__init__()
        self.encoder_first=encoder1()
        self.encoder_second=encoder2(in_channel)
        self.sub_channel1=sub_channel_attention(channels=128,kernel_size=3,ratio=7)
        self.sub_channel2=sub_channel_attention(channels=256,kernel_size=3,ratio=7)
        self.sub_channel3=sub_channel_attention(channels=512,kernel_size=3,ratio=7)
        self.sub_channel4=sub_channel_attention(channels=1024,kernel_size=3,ratio=7)
        self.decoder4=DoubleConv(2048,1024,512)
        self.decoder3=DoubleConv(1024,512,256)
        self.decoder2=DoubleConv(512,256,128)
        self.decoder1=DoubleConv(256,128,64)
        self.conv=nn.Conv2d(64,out_channel,1)

        self.final_conv1 = nn.ConvTranspose2d(64, 32, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self,x):
        skip_connection1,x1,y1=self.encoder_first(x)  # x1为bottlenek后得到的结果，y1为传入encoder部分的结果
        skip_connection2,x2=self.encoder_second(x,y1,skip_connection1[0])  # 传入原图，encoder的，还有第一层的结果
        skip_connection2[3]=self.sub_channel4(skip_connection2[3],x2,0)
        x=x1+x2
        d4=torch.cat([skip_connection1[3],skip_connection2[3],x],dim=1)
        d4=self.decoder4(d4)
        d4=nn.Upsample(scale_factor=2,mode='bilinear')(d4)
        skip_connection2[2]=self.sub_channel3(skip_connection2[2],d4,1)
        d3=torch.cat([skip_connection2[2],skip_connection1[2],d4],dim=1)
        d3=self.decoder3(d3)
        d3 = nn.Upsample(scale_factor=2, mode='bilinear')(d3)
        skip_connection2[1] = self.sub_channel2(skip_connection2[1], d3,2)
        d2=torch.cat([skip_connection1[1],skip_connection2[1],d3],dim=1)
        d2=self.decoder2(d2)
        d2 = nn.Upsample(scale_factor=2, mode='bilinear')(d2)
        skip_connection2[0] = self.sub_channel1(skip_connection2[0], d2,3)
        d1=torch.cat([skip_connection1[0],skip_connection2[0],d2],dim=1)
        d1=self.decoder1(d1)
        d1=nn.Upsample(scale_factor=2,mode='bilinear')(d1)
        out1 = self.final_conv1(d1)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)
        return out


x = torch.randn(6, 3, 224, 224)
model = v1(in_channel=3, out_channel=1)
preds = model(x)
print(preds.shape)
print(x.shape)

