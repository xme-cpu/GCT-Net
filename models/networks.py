import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

import functools
from einops import rearrange

import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d

# ADD these imports at the TOP of networks.py
import timm
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d, AttentionGate, CoordAtt, GlobalContextBlock

###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


# ADD these 4 classes to the END of networks.py, right BEFORE the define_G function

class ConvNeXt_Backbone(nn.Module):
    """
    Loads a ConvNeXt backbone and returns multi-scale features.
    """

    def __init__(self, model_name='convnext_nano', pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[0, 1, 2, 3]  # C2(1/4), C3(1/8), C4(1/16), C5(1/32)
        )
        self.feature_channels = self.backbone.feature_info.channels()
        # e.g., [96, 192, 384, 768] for convnext_tiny

    def forward(self, x):
        features = self.backbone(x)
        return features, self.feature_channels


class G_BiFPN_Decoder(nn.Module):
    """
    Full Gated Bi-directional Feature Pyramid Network (BiFPN)
    Includes:
    - Top-Down Path
    - Bottom-Up Path
    - Gated Fusion (your innovation)
    """

    def __init__(self, feature_channels, bifpn_dim=256, use_gate=True, epsilon=1e-4):
        super().__init__()

        self.use_gate = use_gate
        self.epsilon = epsilon

        C2_ch, C3_ch, C4_ch, C5_ch = feature_channels

        # ---------- 通道统一 ----------
        self.lat_c2 = nn.Conv2d(C2_ch, bifpn_dim, 1)
        self.lat_c3 = nn.Conv2d(C3_ch, bifpn_dim, 1)
        self.lat_c4 = nn.Conv2d(C4_ch, bifpn_dim, 1)
        self.lat_c5 = nn.Conv2d(C5_ch, bifpn_dim, 1)

        # ---------- Gate（只有开启才用）----------
        if self.use_gate:
            self.gate_c4 = AttentionGate(F_g=bifpn_dim, F_l=C4_ch, F_int=C4_ch // 2)
            self.gate_c3 = AttentionGate(F_g=bifpn_dim, F_l=C3_ch, F_int=C3_ch // 2)
            self.gate_c2 = AttentionGate(F_g=bifpn_dim, F_l=C2_ch, F_int=C2_ch // 2)

        # ---------- 可学习权重 ----------
        self.w1 = nn.Parameter(torch.ones(2, 3))
        self.w2 = nn.Parameter(torch.ones(3, 3))

        # ---------- 卷积 ----------
        self.conv_p4_td = nn.Conv2d(bifpn_dim, bifpn_dim, 3, padding=1)
        self.conv_p3_td = nn.Conv2d(bifpn_dim, bifpn_dim, 3, padding=1)
        self.conv_p2_td = nn.Conv2d(bifpn_dim, bifpn_dim, 3, padding=1)

        self.conv_p3_out = nn.Conv2d(bifpn_dim, bifpn_dim, 3, padding=1)
        self.conv_p4_out = nn.Conv2d(bifpn_dim, bifpn_dim, 3, padding=1)
        self.conv_p5_out = nn.Conv2d(bifpn_dim, bifpn_dim, 3, padding=1)

    def forward(self, backbone_feats, c5_input):

        C2, C3, C4 = backbone_feats

        P5 = self.lat_c5(c5_input)

        # ===== Gate开关 =====
        if self.use_gate:
            P5_up_4 = F.interpolate(P5, size=C4.shape[-2:], mode='bilinear', align_corners=False)
            C4 = self.gate_c4(P5_up_4, C4)

            P5_up_3 = F.interpolate(P5, size=C3.shape[-2:], mode='bilinear', align_corners=False)
            C3 = self.gate_c3(P5_up_3, C3)

            P5_up_2 = F.interpolate(P5, size=C2.shape[-2:], mode='bilinear', align_corners=False)
            C2 = self.gate_c2(P5_up_2, C2)

        P4_in = self.lat_c4(C4)
        P3_in = self.lat_c3(C3)
        P2_in = self.lat_c2(C2)

        # ---------- Top-Down ----------
        P4_td = self._wsum([P4_in, F.interpolate(P5, size=P4_in.shape[-2:])], self.w1[:, 0])
        P4_td = self.conv_p4_td(P4_td)

        P3_td = self._wsum([P3_in, F.interpolate(P4_td, size=P3_in.shape[-2:])], self.w1[:, 1])
        P3_td = self.conv_p3_td(P3_td)

        P2_td = self._wsum([P2_in, F.interpolate(P3_td, size=P2_in.shape[-2:])], self.w1[:, 2])
        P2_td = self.conv_p2_td(P2_td)

        # ---------- Bottom-Up ----------
        P3_out = self._wsum([P3_in, P3_td, F.max_pool2d(P2_td, 2)], self.w2[:, 0])
        P3_out = self.conv_p3_out(P3_out)

        P4_out = self._wsum([P4_in, P4_td, F.max_pool2d(P3_out, 2)], self.w2[:, 1])
        P4_out = self.conv_p4_out(P4_out)

        P5_out = self._wsum([P5, F.max_pool2d(P4_out, 2)], self.w2[:2, 2])
        P5_out = self.conv_p5_out(P5_out)

        return P2_td

    def _wsum(self, inputs, weights):
        weights = F.relu(weights)
        weights = weights / (weights.sum() + self.epsilon)

        out = 0
        for i in range(len(inputs)):
            out += weights[i] * inputs[i]
        return out

class CDEM_Head(nn.Module):
    """
    消融模块: CDEM
    开关: mode='cdem' (协同差分), mode='abs_diff' (绝对差值)
    """

    def __init__(self, in_channels, num_classes=2, mode='cdem'):
        super().__init__()
        self.mode = mode  # <--- 这里的开关

        if self.mode == 'cdem':
            # 创新点: 可学习差分 + 协同注意力
            self.diff_conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True)
            )
            self.coord_att = CoordAtt(in_channels, in_channels)
            self.classifier = nn.Conv2d(in_channels, num_classes, 1)

        else:
            # Baseline: 绝对差值 + 简单卷积
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, num_classes, 1)
            )

    def forward(self, f1, f2):
        if self.mode == 'cdem':
            x = torch.cat([f1, f2], dim=1)
            x = self.diff_conv(x)
            x = self.coord_att(x)
            out = self.classifier(x)
        else:
            # 传统方法
            out = self.classifier(torch.abs(f1 - f2))

        return F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)


class Hybrid_BIT(nn.Module):
    """
    主模型类，接收所有开关
    """

    def __init__(self, input_nc=3, output_nc=2,
                 token_len=4, enc_depth=1, dec_depth=8,
                 dim_head=64, decoder_dim_head=64,
                 bifpn_dim=256,
                 use_strm=True, use_gate=True, diff_mode='cdem'):
        super().__init__()
        self.use_strm = use_strm  # <--- STRM 开关

        # 1. Backbone
        self.backbone = ConvNeXt_Backbone()
        backbone_channels = self.backbone.feature_channels
        c5_channels = backbone_channels[-1]

        bit_dim = 32

        # 2. Deep Feature Processing (STRM vs Baseline)
        if self.use_strm:
            # 创新点: GCNet + BIT
            self.gcnet = GlobalContextBlock(in_channels=c5_channels)
            self.bit_pre_conv = nn.Conv2d(c5_channels, bit_dim, kernel_size=1)

            self.token_len = token_len
            self.conv_a = nn.Conv2d(bit_dim, self.token_len, kernel_size=1, padding=0, bias=False)
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, bit_dim))

            self.transformer_encoder = Transformer(dim=bit_dim, depth=enc_depth, heads=8,
                                                   dim_head=dim_head, mlp_dim=bit_dim * 2, dropout=0)
            self.transformer_decoder = TransformerDecoder(dim=bit_dim, depth=dec_depth, heads=8,
                                                          dim_head=decoder_dim_head, mlp_dim=bit_dim * 2,
                                                          dropout=0, softmax=True)
        else:
            # Baseline: 仅降维
            self.proj_c5 = nn.Conv2d(c5_channels, bit_dim, 1)

        # 3. Neck
        g_bifpn_channels = backbone_channels[:-1] + [bit_dim]
        self.g_bifpn_decoder = G_BiFPN_Decoder(
            feature_channels=g_bifpn_channels,
            bifpn_dim=bifpn_dim,
            use_gate=use_gate  # 传入门控开关
        )

        # 4. Head
        self.head = CDEM_Head(in_channels=bifpn_dim, num_classes=output_nc, mode=diff_mode)  # 传入差分开关

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def forward(self, x1, x2):
        # Backbone
        ft1, _ = self.backbone(x1)
        ft2, _ = self.backbone(x2)
        c5_1, c5_2 = ft1[-1], ft2[-1]

        # STRM vs Baseline
        if self.use_strm:
            # GCNet
            c5_1 = self.gcnet(c5_1)
            c5_2 = self.gcnet(c5_2)
            # BIT
            x_bit_1 = self.bit_pre_conv(c5_1)
            x_bit_2 = self.bit_pre_conv(c5_2)

            token1 = self._forward_semantic_tokens(x_bit_1)
            token2 = self._forward_semantic_tokens(x_bit_2)
            tokens_cat = torch.cat([token1, token2], dim=1)
            tokens_cat += self.pos_embedding
            tokens_enc = self.transformer_encoder(tokens_cat)
            t1, t2 = tokens_enc.chunk(2, dim=1)

            x_new_1 = self._forward_transformer_decoder(x_bit_1, t1)
            x_new_2 = self._forward_transformer_decoder(x_bit_2, t2)
        else:
            # Baseline
            x_new_1 = self.proj_c5(c5_1)
            x_new_2 = self.proj_c5(c5_2)

        # Neck
        f_final_1 = self.g_bifpn_decoder(ft1[:-1], x_new_1)
        f_final_2 = self.g_bifpn_decoder(ft2[:-1], x_new_2)

        # Head
        return self.head(f_final_1, f_final_2)


# --------------------------------------------------------------------------
# 3. 定义入口函数 (手动开关在这里修改！)
# --------------------------------------------------------------------------

def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'base_resnet18':
        net = ResNet(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'hybrid_bit':

        # ================================================================
        # 【兄弟，做消融实验改这里】
        # 把不需要的模式注释掉，保留你想跑的那一行
        # ================================================================

        # 实验0: Baseline (纯ConvNeXt + FPN + 绝对差值)
        # experiment_mode = 'baseline'

        # 实验1: 验证 STRM (ConvNeXt + STRM + FPN + 绝对差值)
        # experiment_mode = 'add_strm'

        # 实验2: 验证 G-BiFPN (ConvNeXt + STRM + G-BiFPN + 绝对差值)
        # experiment_mode = 'add_gate'

        # 实验3: 完整模型 GCT-Net (全开)
        experiment_mode = 'full'

        # ================================================================

        print(f"==================================================")
        print(f"正在加载 Hybrid_BIT 模型，当前消融模式: 【{experiment_mode}】")
        print(f"==================================================")

        if experiment_mode == 'baseline':
            use_strm, use_gate, diff_mode = False, False, 'abs_diff'
        elif experiment_mode == 'add_strm':
            use_strm, use_gate, diff_mode = True, False, 'abs_diff'
        elif experiment_mode == 'add_gate':
            use_strm, use_gate, diff_mode = True, True, 'abs_diff'
        else:  # full
            use_strm, use_gate, diff_mode = True, True, 'cdem'

        net = Hybrid_BIT(input_nc=3, output_nc=args.n_class,
                         token_len=4, enc_depth=1, dec_depth=8,
                         dim_head=64, decoder_dim_head=64,
                         bifpn_dim=256,
                         use_strm=use_strm,
                         use_gate=use_gate,
                         diff_mode=diff_mode)

    elif args.net_G == 'base_transformer_pos_s4':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                               with_pos='learned')
    elif args.net_G == 'base_transformer_pos_s4_dd8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                               with_pos='learned', enc_depth=1, dec_depth=8)
    elif args.net_G == 'base_transformer_pos_s4_dd8_dedim8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                               with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)

    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x


class BASE_Transformer(ResNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_Transformer, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos is 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode is 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode is 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        # feature differencing
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        # forward small cnn
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x


