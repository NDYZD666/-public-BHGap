# --------------------------------------------------------
# References:
# timm: https://github.com/huggingface/pytorch-image-models
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch

from timm.models.vision_transformer import VisionTransformer
import torch.nn as nn

class VisionTransformer2(VisionTransformer):
    def __init__(self, global_pool=False,
                  n_seq=196, 
                  n_progr=2, 
                  prompt_depth = 12, 
                  new_promt = 1, 
                  generate_prompt = 1,
                  n_frames=16, **kwargs):
        super(VisionTransformer2, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

        self.n_seq = n_seq  
        self.n_progr = n_progr  
        self.n_frames = n_frames  
        self.prompt_depth = prompt_depth
        self.new_prompt = new_promt
        self.generate_prompt = generate_prompt

        self.latent_dim = 128
       

        self.learnable_prompts_init = nn.Parameter(
            torch.randn(self.n_progr * self.prompt_depth, 768) * (768 **-0.5))

        self.learnable_prompts_progr = nn.ParameterList(
            [nn.Parameter(torch.randn(self.new_prompt, 768) * (768 **-0.5)) for i in range(self.new_prompt * self.prompt_depth)])


        self.all_gate = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for i in range(len(self.blocks))])
   


    def forward_block_pre(self, ii, x, B):


        if ii == 0:  
            
            x = self.patch_embed(x)


            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            

            x = torch.cat((cls_tokens, x, self.learnable_prompts_init.expand(B, -1, -1)), dim=1)
            

            x = x + self.pos_embed
           

            x = self.pos_drop(x)


        x = self.blocks[ii](x)
        return x



    def forward_block_post(self, ii, x, p_t, generate_prompt, B):

        
        if ii < 12:
            p_t_repeat = p_t.repeat(16, 1, 1)
            g_p_repeat = generate_prompt.repeat(16, 1, 1)
            prompts_progr = self.learnable_prompts_progr[ii].expand(B, -1, -1)
            start_index_1 = self.n_seq+1+ii*self.new_prompt
            end_index_1 = self.n_seq+1+(ii+1)*self.new_prompt

            start_index_2 = self.n_seq+1+self.prompt_depth*self.new_prompt+ii*self.generate_prompt
            end_index_2 = self.n_seq+1+self.prompt_depth*self.new_prompt+(ii+1)*self.generate_prompt

            x_1 = x[:,0:start_index_1,:]
            x_2 = x[:,start_index_1:end_index_1,:] + prompts_progr
            x_3 = x[:,end_index_1:start_index_2,:]
            x_4 = x[:,start_index_2:end_index_2,:] + p_t_repeat
            x_5 = x[:,end_index_2:,:]
            x = torch.cat((x_1, x_2, x_3, x_4, x_5), dim=1)
        x = x + nn.functional.tanh(self.all_gate[ii])* g_p_repeat


        
        return x #outcome

 
    # borrow from timm
    def forward(self, x, ret_feature=False):

        x = self.forward_features(x)

        
        feature = x

        if getattr(self, 'head_dist', None) is not None:

            x, x_dist = self.head(x[0]), self.head_dist(x[1])  
            if self.training and not torch.jit.is_scripting():
 
                
                return x, x_dist
            else:
               
                return (x + x_dist) / 2
  
        else:
            x = self.head(x)



        if ret_feature:
            return x, feature
        
        else:
            return x


# setup model archs
VIT_KWARGS_BASE = dict(mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))

VIT_KWARGS_PRESETS = {
    'tiny': dict(patch_size=16, embed_dim=192, depth=12, num_heads=3),
    'small': dict(patch_size=16, embed_dim=384, depth=12, num_heads=6),
    'base': dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    'large': dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    'huge': dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
    'giant': dict(patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11),
    'gigantic': dict(patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=64/13),
}

def create_vit_model(preset=None, creator=None, **kwargs):
    preset = 'base' if preset is None else preset.lower()
    all_kwargs = dict()
    all_kwargs.update(VIT_KWARGS_BASE)
    all_kwargs.update(VIT_KWARGS_PRESETS[preset])
    all_kwargs.update(kwargs)
    if creator is None:
        creator = VisionTransformer2
    return creator(**all_kwargs)

#vit_tiny_patch16 = partial(create_vit_model, preset='tiny')
#vit_small_patch16 = partial(create_vit_model, preset='small')
vit_base_patch16 = partial(create_vit_model, preset='base')
#vit_large_patch16 = partial(create_vit_model, preset='large')
#vit_huge_patch14 = partial(create_vit_model, preset='huge')
#vit_giant_patch14 = partial(create_vit_model, preset='giant')
#vit_gigantic_patch14 = partial(create_vit_model, preset='gigantic')
