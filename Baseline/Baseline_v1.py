import torch
import torch.nn as nn
from torchvision import models as resnet_model
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from thop import profile

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # for _ in range(depth):

        self.layers.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)


class Bottleneck(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Bottleneck,self).__init__()
        self.bottleneck=DoubleConv(in_channel,out_channel)

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
        self.bottleneck = Bottleneck(in_channel=512, out_channel=1024)
    def forward(self,x):
        skip_connection=[]
        x=self.firstconv(x)
        x=self.firstbn(x)
        x=self.firstrelu(x)
        x=self.encoder1(x)
        skip_connection.append(x)
        x=self.encoder2(x)
        skip_connection.append(x)
        x=self.encoder3(x)
        skip_connection.append(x)
        x=self.encoder4(x)
        skip_connection.append(x)
        x=self.bottleneck(x)

        return skip_connection,x


class encoder2(nn.Module):
    # def __init__(self):
    #     super(encoder,self).__init__()

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)   # patch的数量
        patch_dim = channels * patch_height * patch_width       # patch的维度
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.channelTrans = nn.Conv2d(in_channels=65, out_channels=512, kernel_size=1, padding=0)

        transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        self.patch_embed = transformer.patch_embed
        self.transformers = nn.ModuleList(
            [transformer.blocks[i] for i in range(12)]
        )

    def forward(self, x):
        x_vit = x
        x_vit = self.to_patch_embedding(x_vit)
        b, n, _ = x_vit.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x_vit = torch.cat((cls_tokens, x_vit), dim=1)
        x_vit += self.pos_embedding[:, :(n + 1)]
        x_vit = self.dropout(x_vit)

        vit_layerInfo = []
        for i in range(4):  # 设置深度的地方[6, 64+1, dim=196]
            x_vit = self.transformer(x_vit)
            vit_layerInfo.append(x_vit)


        # a = vit_layerInfo[0]
        # a = a.view(6, 65, 14, 14)
        #
        # # 后续的三步处理，目前不需要
        # x_vit = x_vit.mean(dim = 1) if self.pool == 'mean' else x_vit[:, 0]
        # x_vit = self.to_latent(x_vit)
        # x_vit = self.mlp_head(x_vit)
        #
        # emb = self.patch_embed(x)
        # for i in range(12):
        #     emb = self.transformers[i](emb)
        # feature_tf = emb.permute(0, 2, 1)
        # feature_tf = feature_tf.view(b, 192, 14, 14)
        # feature_tf = self.conv_seq_img(feature_tf)

        return vit_layerInfo

class Baseline(nn.Module):
    def __init__(self,in_channel,out_channel,features=[64, 128, 256, 512]):
        super(Baseline, self).__init__()
        self.encoder1=encoder1()
        self.encoder2=encoder2(image_size=224, patch_size=28, num_classes=2, dim=196, depth=6, heads=16, mlp_dim=2048)
        self.bottleneck = Bottleneck(in_channel=65, out_channel=1024)
        self.vitLayer_UpConv = nn.ConvTranspose2d(65, 65, kernel_size=2, stride=2)
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2 + 65 + feature, feature * 2, kernel_size=3, stride=1,padding=1
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.upsample1=nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.upsample2=nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.upsample3=nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.upsample4=nn.ConvTranspose2d(128,128 , kernel_size=2, stride=2)
        self.upsample5=nn.ConvTranspose2d(64,64 , kernel_size=2, stride=2)


        self.final_conv1 = nn.ConvTranspose2d(64, 32, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, out_channel, 3, padding=1)

    def forward(self,x):
        skip_connection1,x1=self.encoder1(x)
        vit_layerInfo=self.encoder2(x)

        vit_layerInfo = vit_layerInfo[::-1]# 翻转，呈正序。0表示第四层...3表示第一层
        b,_,_=vit_layerInfo[0].shape
        v = vit_layerInfo[0].view(b, 65, 14, 14)
        v1=self.bottleneck(v)

        x=v1+x1
        v=self.vitLayer_UpConv(v)
        x=self.upsample1(x)
        skip_connection1[3]=self.upsample2(skip_connection1[3])
        x = torch.cat([x, skip_connection1[3], v], dim=1)
        x = self.ups[0](x)
        x = self.ups[1](x)

        v = vit_layerInfo[1].view(b, 65, 14, 14)
        v = self.vitLayer_UpConv(v)
        v = self.vitLayer_UpConv(v)
        x = self.upsample2(x)
        skip_connection1[2]=self.upsample3(skip_connection1[2])
        x = torch.cat([x, skip_connection1[2], v], dim=1)
        x = self.ups[2](x)
        x = self.ups[3](x)

        v = vit_layerInfo[2].view(b, 65, 14, 14)
        v = self.vitLayer_UpConv(v)
        v = self.vitLayer_UpConv(v)
        v = self.vitLayer_UpConv(v)
        x = self.upsample3(x)
        skip_connection1[1] = self.upsample4(skip_connection1[1])
        x = torch.cat([x, skip_connection1[1], v], dim=1)
        x = self.ups[4](x)
        x = self.ups[5](x)

        v = vit_layerInfo[3].view(b, 65, 14, 14)
        v = self.vitLayer_UpConv(v)
        v = self.vitLayer_UpConv(v)
        v = self.vitLayer_UpConv(v)
        v = self.vitLayer_UpConv(v)
        x = self.upsample4(x)
        skip_connection1[0] = self.upsample5(skip_connection1[0])
        x = torch.cat([x, skip_connection1[0], v], dim=1)
        x = self.ups[6](x)
        x = self.ups[7](x)

        out1 = self.final_conv1(x)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out

x=torch.randn(6,3,224,224)
model=Baseline(in_channel=3,out_channel=1)
preds=model(x)
print(preds.shape)
print(x.shape)


