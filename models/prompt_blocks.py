import torch
import torch.nn as nn
import torch.nn.functional as F
from models.selective_scan_interface import mamba_inner_fn,selective_scan_fn
from einops import rearrange


class Img_PatchLevelConvBlock(nn.Module):
    def __init__(self, in_dim=221,d_model = 768):
        super().__init__()
        self.d_model = d_model
        self.in_dim = in_dim
        self.linear_up = nn.Linear(768,1024)
        self.linear_down = nn.Linear(1024,768)
        self.pointwise_conv_in = nn.Sequential(nn.Conv2d(in_channels=self.in_dim,
                                           out_channels=self.in_dim,
                                           kernel_size = 1),
                                           nn.BatchNorm2d(self.in_dim),
                                           nn.SiLU())
        self.pointwise_conv_out = nn.Sequential(nn.Conv2d(in_channels=self.in_dim,
                                           out_channels=self.in_dim,
                                           groups=self.in_dim,
                                           kernel_size = 3,
                                           padding=1),
                                           nn.BatchNorm2d(self.in_dim),
                                           nn.SiLU())
    def forward(self,B,x):
        x = self.linear_up(x)  
        x = x.reshape(B, 32, 32, -1).permute(0, 3, 1, 2).contiguous()
        x = self.pointwise_conv_in(x)
        x = self.pointwise_conv_out(x)
        x = x.permute(0, 2, 3, 1).reshape(B, self.in_dim, -1)
        x = self.linear_down(x)
        return x
        
class Audio_PatchLevelConvBlock(nn.Module):
    def __init__(self, in_dim=281,d_model = 768):
        super().__init__()
        self.d_model = d_model
        self.in_dim = in_dim
        self.linear_up = nn.Linear(768,1024)
        self.linear_down = nn.Linear(1024,768)
        self.pointwise_conv_in = nn.Sequential(nn.Conv2d(in_channels=self.in_dim,
                                           out_channels=self.in_dim,
                                           kernel_size = 1),
                                           nn.BatchNorm2d(self.in_dim),
                                           nn.SiLU())
        self.pointwise_conv_out = nn.Sequential(nn.Conv2d(in_channels=self.in_dim,
                                           out_channels=self.in_dim,
                                           groups=self.in_dim,
                                           kernel_size = 3,
                                           padding=1),
                                           nn.BatchNorm2d(self.in_dim),
                                           nn.SiLU())
    def forward(self,B,x):
        x = self.linear_up(x)
        x = x.reshape(B, 32, 32, -1).permute(0, 3, 1, 2).contiguous()
        x = self.pointwise_conv_in(x)
        x = self.pointwise_conv_out(x)
        x = x.permute(0, 2, 3, 1).reshape(B, self.in_dim, -1)
        x = self.linear_down(x)
        return x                                     
        
class SSMBlock(nn.Module):
    """Custom SSM module based on selective_scan interface, 
        referring to the official Mamba implementation, 
        thanks for their work
    """
    def __init__(self, hidden_dim = 768, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = hidden_dim
        self.d_state = d_state
        self.d_conv = d_conv
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.x_proj = nn.Linear(hidden_dim, d_state * 2 + d_state)  
        
        self.img_conv = Img_PatchLevelConvBlock()
        self.audio_conv = Audio_PatchLevelConvBlock()   
        
        # S4D
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(A_log.repeat(hidden_dim, 1))
        self.A_log._no_weight_decay = True
        
        # D
        self.D = nn.Parameter(torch.ones(hidden_dim))
        self.D._no_weight_decay = True
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dt_proj = nn.Linear(d_state, hidden_dim)
        
    def forward(self, image, audio):
        """
        输入: [B, L, D]
        输出: [B, L, D]
        """
        image_feat = image.clone()
        audio_feat = audio.clone()

        B, L, D = image_feat.shape
        batch_size = audio_feat.shape[0]

        image_feat = image_feat.view(B//16, 16, -1, D)
 
        image_feat = torch.mean(image_feat, dim=1)  
        # print('image_feat.shape: ', image_feat.shape)

        image_feat = self.img_conv(batch_size, image_feat)
        audio_feat = self.audio_conv(batch_size, audio_feat)
        # print('image_feat_conv.shape: ', image_feat.shape)
        # print('audio_feat_conv.shape: ', audio_feat.shape)
        conbine = torch.cat((image_feat, audio_feat), dim=1)
        batch, seqlen, dim = conbine.shape
        

        x = conbine.transpose(1, 2)  # [B, D, L]
        z = x.clone()  
        
  
        x_dbl = self.x_proj(x.transpose(1, 2).reshape(-1, dim))  # [(B*L), 48]
        dt, B_param, C = torch.split(x_dbl, [self.d_state, self.d_state, self.d_state], dim=-1)
        

        dt = self.dt_proj.weight @ dt.t()  # [d_model, (b*l)]

        dt = rearrange(dt, "d (b l) -> b d l", b=batch, l=seqlen)
        

        B_param = rearrange(B_param, "(b l) dstate -> b dstate l", b=batch, l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", b=batch, l=seqlen).contiguous()
        
 
        A = -torch.exp(self.A_log)  # [D, d_state]
        
        y = selective_scan_fn(
            x,           # [B, D, L]
            dt,          # [B, D, L] 
            A,           # [D, d_state]
            B_param,     # [B, d_state, L]
            C,           # [B, d_state, L]
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float() if hasattr(self.dt_proj, 'bias') else None,
            delta_softplus=True
        )
        

        y = y.transpose(1, 2)  # [B, L, D]
        y = self.out_proj(y)

        image_len = image_feat.shape[1]
        audio_len = audio_feat.shape[1]
        
        
        assert y.shape[1] == image_len + audio_len, "The sequence length does not match"
        
        y = self.norm(y)
        image_output = y[:, :image_len, :]
        audio_output = y[:, image_len:, :]
        

        image_output = image_output.mean(dim=1).unsqueeze(1)
        audio_output = audio_output.mean(dim=1).unsqueeze(1)

        return image_output, audio_output
