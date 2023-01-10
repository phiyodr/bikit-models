"""
All code is by Sofia who submitted via https://dacl.ai/.
https://github.com/mpaques269546/codebrim_challenge
"""

import os
import math
import sys

import torch
import torch.nn as nn


####### Decoder:
class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features
    activation = nn.Sigmoid(), nn.Softmax"""
    def __init__(self, embed_dim=384, num_cls=13,  activation=None):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(embed_dim, num_cls)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
      
        self.activation = activation
        self.arch = 'linear_classifier'


    def forward(self, x):
        # linear layer
        x = self.linear(x)
        #activation
        if self.activation: 
            x = self.activation(x)
        return x


####### Encoder:        
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.depth = depth
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        #trunc_normal_(self.pos_embed, std=.02)
        #trunc_normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            #trunc_normal_(m.weight, std=.02)
            torch.nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x, H, W = self.patch_embed(x)  # patch linear embedding
        

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x), W, H

    def forward(self, x):
        x , _, _= self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x, _, _ = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x, _, _ = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def forward_classification(self, x, n=4, avgpool=False):
        intermediate_output = self.get_intermediate_layers(x, n)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        #print('output =', output.shape) # [B, 1536] #1536=4*384=n*(D + int(avgpool)
        if avgpool:
            output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            output = output.reshape(output.shape[0], -1)
        return output


####### Build the model:
def build_from_cp(model, pretrained_weights, checkpoint_key, model_name):
    
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        #print(state_dict.keys())
        if checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]

        #print('pretrained=', state_dict.keys())
        #print('model sate dict=', model.state_dict().keys())

        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # remove `model.` prefix induced by saving models
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            
        # in case different inp size: change pos_embed 
        if "pos_embed" in state_dict.keys():
            pretrained_shape = state_dict['pos_embed'].size()[1]
            model_shape = model.state_dict()['pos_embed'].size()[1]
            if pretrained_shape != model_shape:
                pos_embed = state_dict['pos_embed']
                pos_embed = pos_embed.permute(0, 2, 1)
                pos_embed = nn.functional.interpolate(pos_embed, size=model_shape)
                pos_embed = pos_embed.permute(0, 2, 1)
                state_dict['pos_embed'] = pos_embed
        #ignore linear layer
        #if "linear.bias" in state_dict.keys():
        #del state_dict['linear.weight'], state_dict['linear.bias']

        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

    else:
        print('ERROR: file {0} does not exist'.format(pretrained_weights))
        print('We use random weights.')


def build_encoder( pretrained_weights='', key='',  arch=None, patch_size=8 , avgpool=False, image_size=224, drop_rate=0, trainable=False):
    
    arch_dic = {'vit_tiny':{ 'd_model':384, 'n_heads':3, 'n_layers':12},
      'vit_small':{ 'd_model':384, 'n_heads':6, 'n_layers':12},
      'vit_base':{'d_model':384, 'n_heads':12, 'n_layers':12},
      'vit_large':{ 'd_model':384, 'n_heads':24, 'n_layers':12},}

    if arch in arch_dic.keys():
        d_model = arch_dic[arch]['d_model']
        n_heads = arch_dic[arch]['n_heads']
        n_layers = arch_dic[arch]['n_layers']
        n_cls = 1 #don't care only usefull for classification head
        # image_size=
        model = VisionTransformer(img_size=[image_size], patch_size=patch_size, in_chans=3, num_classes=n_cls, embed_dim=d_model, depth=n_layers,
                 num_heads=n_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm)
        
       

    else:
        print(f"Unknow architecture: {arch}")
        sys.exit(1)

    # load weights to evaluate
    if len(key)>0 and model!=None and len(pretrained_weights)>0:
        build_from_cp(model, pretrained_weights, key, arch)
        print('pretrained weights loaded')

    # params
    # freeze or not weights
    ct, cf =0 ,0
    if trainable:
        print('trainable encoder', key)
        for p in model.parameters():
            p.requires_grad = True
            ct+= p.numel()
    else:
        print('frozen encoder ', key)
        for p in model.parameters():
            p.requires_grad = False
            cf+= p.numel()
    print(f"{arch} adapter built. {ct} trainable params, {cf} frozen params.")
   
    return model


def build_decoder(pretrained_weights, key,   num_cls=2, embed_dim=384*4, image_size=224, activation=None, trainable=False):
    
    model = LinearClassifier( embed_dim=embed_dim, num_cls=num_cls,  activation=activation )
    
    # load weights to evaluate
    if len(key)>0 and model!=None and len(pretrained_weights)>0:
        arch=""
        build_from_cp(model, pretrained_weights, key, arch)
        print('pretrained weights loaded')

    # params
    # freeze or not weights
    ct, cf =0 ,0
    if trainable:
        print('trainable decoder', key)
        for p in model.parameters():
            p.requires_grad = True
            ct+= p.numel()
    else:
        print('frozen decoder ', key)
        for p in model.parameters():
            p.requires_grad = False
            cf+= p.numel()
    print(f"{key} adapter built. {ct} trainable params, {cf} frozen params.")
    return model


       

class Ensemble(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        trainable_encoder=False,
        trainable_decoder=False,
        n_last_blocks=4,
        avgpool_patchtokens=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trainable_encoder = trainable_encoder
        self.trainable_decoder = trainable_decoder
        self.n_last_blocks= n_last_blocks
        self.avgpool_patchtokens = avgpool_patchtokens


    def forward(self, inp, return_features=False):
        ## encoder
        # final_features = [B, D]
        if self.trainable_encoder:
            features = self.encoder.forward_classification(inp, n=self.n_last_blocks, avgpool=self.avgpool_patchtokens)
        else:
            with torch.no_grad():
                features = self.encoder.forward_classification(inp, n=self.n_last_blocks, avgpool=self.avgpool_patchtokens)
                features = features.detach()
        
        ## decoder
        # output = [B, num_cls]     
        if self.trainable_decoder:
            output = self.decoder.forward(features )
        else:
            with torch.no_grad():
                output = self.decoder.forward(features)
        if return_features:
            return [features, output]
       
        return  output



def build_vistra(cp_path, img_size=224, num_cls=6, quantized=True, activation=None):
    if quantized:
        weights_encoder_decoder = ''
    else:
        weights_encoder_decoder = cp_path

    encoder = build_encoder(weights_encoder_decoder, arch='vit_small',  key='encoder', image_size=img_size)
    n_last_blocks = 4
    avgpool_patchtokens = False
    embed_dim = encoder.embed_dim * (n_last_blocks + int(avgpool_patchtokens))

    decoder = build_decoder(weights_encoder_decoder, key='decoder', num_cls=num_cls, 
                            embed_dim=embed_dim, image_size=img_size, activation=None) # Activation in decoder: nn.Sigmoid()

    model = Ensemble(encoder, decoder,n_last_blocks= n_last_blocks, avgpool_patchtokens=avgpool_patchtokens)

    cat_to_name = None
   
    if quantized:
        print('quantization')
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d, 
                                                            torch.nn.Dropout, torch.nn.GELU,
                                                            torch.nn.ReLU, torch.nn.LayerNorm}, 
                                                            dtype=torch.qint8)
        
        cp = torch.load(cp_path, map_location="cpu")
        #print('state dict:', state_dict.keys())
        msg = model.load_state_dict(cp['state_dict'], strict=False)
        model_summary = 'The model was instantiated from {} and loaded with msg: {} with the following arguments:\n'.format(cp_path, msg)
        for key, value in cp.items():
            if key != 'state_dict':
                model_summary += f"{key}: {value}\n"
        print('=====Model summary=====')
        print(model_summary)
        cat_to_name = cp['cat_to_name']
    return model, cat_to_name


if __name__ == '__main__':
    # Quick check
    model, cat_to_name = build_vistra(cp_path= 'models/checkpoints/codebrim-classif-balanced/codebrim-classif-balanced_ViT_s8_manipulated.pth')

    # cp_path = 'models/checkpoints/codebrim-classif-balanced/codebrim-classif-balanced_ViT_s8_1.pth'
    # cp = {}
    # cp['state_dict'] = torch.load(cp_path, map_location="cpu")
    # cp['dataset'] = 'codebrim-classif-balanced'
    # cp['img_size'] = 224
    # cp['num_class'] = 6
    # cp['cat_to_name'] = {0:'NoDamage', 1:'Crack', 2:'Spalling', 3:'Efflorescence', 4:'BarsExposed', 5:'Rust'}

    # model_summary = 'The model was instantiated from {} with the following arguments:\n'.format(cp_path)
    # for key, value in cp.items():
    #     if key != 'state_dict':
    #         model_summary += f"{key}: {value}\n"
    # print(model_summary)
    # torch.save(cp, ('models/checkpoints/codebrim-classif-balanced/codebrim-classif-balanced_ViT_s8_manipulated.pth'))   

    # cp_path = 'models/checkpoints/codebrim-classif-balanced/codebrim-classif-balanced_ViT_s8_0.pth'
    # cp = torch.load(cp_path, map_location="cpu")
    # cp = {}
    # cp['state_dict'] = torch.load(cp_path, map_location="cpu")
    # cp['dataset'] = 'codebrim-classif-balanced'
    # cp['img_size'] = 224
    # cp['num_class'] = 6
    # cp['cat_to_name'] = {0:'NoDamage', 1:'Crack', 2:'Spalling', 3:'Efflorescence', 4:'BarsExposed', 5:'Rust'}

    # model_summary = 'The model was instantiated from {} with the following arguments:\n'.format(cp_path)
    # for key in cp['state_dict'].keys():
    #     if key != 'state_dict':
    #         model_summary += f"{key}\n"
    # print(model_summary)