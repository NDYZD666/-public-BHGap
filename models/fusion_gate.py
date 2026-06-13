import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba,Mamba2

class MSC2F(nn.Module):
    def __init__(self, 
                 video_dim=768,
                 audio_dim=768,
                 hidden_dim=512,
                 num_frames=16,
                 num_blocks=3,
                 num_classes=7,
                 fusion_type='transformer'):
        super().__init__()
        
        self.fusion_type = fusion_type
        

        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )


        self.align_module = LowRankAdversarialAlign(hidden_dim, rank=8)

        

        self.impl_align = ImplicitAlign(hidden_dim)
        

        self.gate_res = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        

        if self.fusion_type == 'mamba':

            self.mamba_blocks = nn.ModuleList([
                MambaBlock(hidden_dim) for _ in range(num_blocks)
            ])
        elif self.fusion_type == 'transformer':

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.transformer_fusion = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=num_blocks
            )

        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.align_module.U)
        nn.init.xavier_uniform_(self.align_module.V)
        nn.init.normal_(self.impl_align.alpha, mean=0, std=0.01)

    def forward(self, video, audio):
        B, F, S1, Dv = video.shape
        

        video_feat = self.video_proj(video).view(B, F*S1, -1)  # [B, T_v, D]
        audio_feat = self.audio_proj(audio)  # [B, T_a, D]
        

        adv_loss = torch.tensor(0.0).to(video_feat.device)
        mmd_loss = torch.tensor(0.0).to(video_feat.device)
        

        aligned_audio, adv_loss, gamma = self.align_module(video_feat, audio_feat)


        refined_audio, mmd_loss = self.impl_align(video_feat, aligned_audio)

        

        aligned_audio_feat = torch.bmm(gamma, audio_feat)  
            
        gate = self.gate_res(aligned_audio_feat)
        final_audio = gate * aligned_audio_feat + (1 - gate) * refined_audio

        

        combined = torch.cat([video_feat, final_audio], dim=1)  # [B, T_total, D]
        
 
        if self.fusion_type == 'mamba':
            # Mamba
            for block in self.mamba_blocks:
                combined = block(combined)

            cls_output = self.classifier(combined.mean(1))
            
        elif self.fusion_type == 'transformer':
            # Transformer
            combined = self.transformer_fusion(combined)

            cls_output = self.classifier(combined.mean(1))
            
        return cls_output, adv_loss, mmd_loss

class MambaBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mamba = Mamba(
            d_model=hidden_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.mamba(x)
        x = self.norm(x)
        return x

class LowRankAdversarialAlign(nn.Module):
    def __init__(self, hidden_dim, rank=8):
        super().__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.randn(1, 4000, rank))
        self.V = nn.Parameter(torch.randn(1, 4000, rank))
        
   
        self.D = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
        

        self.G = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        

        self.lambda_fm = 0.1
        

        self.grl = GradientReversal(alpha=1.0)
        
    def forward(self, video_feat, audio_feat):
        B, Tv, D = video_feat.shape
        Ta = audio_feat.shape[1]
        

        U = self.U.expand(B, -1, -1)[:, :Tv, :]
        V = self.V.expand(B, -1, -1)[:, :Ta, :]
        gamma = torch.bmm(U, V.transpose(1,2))
        

        aligned_audio_raw = torch.bmm(gamma, audio_feat)
        
 
        aligned_audio = self.G(aligned_audio_raw)
        

        if self.training:
            # Wasserstein GAN with gradient penalty

            real_logits = self.D(video_feat)
            
 
            fake_logits = self.D(self.grl(aligned_audio))
            
            # WGAN Loss
            d_loss = -torch.mean(real_logits) + torch.mean(fake_logits)
            
            # Improve g_loss to avoid gradient cancellation
            # We take the value of d_loss as the coefficient of g_loss, but cut off its gradient flow
            g_loss = -torch.mean(fake_logits) * torch.abs(d_loss.detach())
            
            try:
                epsilon = torch.rand(B, Tv, 1, device=video_feat.device)
                interpolated = epsilon * video_feat + (1 - epsilon) * aligned_audio.detach()
                interpolated.requires_grad_(True)
                
                mixed_logits = self.D(interpolated)
                
                if interpolated.requires_grad and mixed_logits.requires_grad:
                    gradients = torch.autograd.grad(
                        outputs=mixed_logits,
                        inputs=interpolated,
                        grad_outputs=torch.ones_like(mixed_logits),
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    
                    if gradients is not None:
                        gradients = gradients.view(B, -1)
                        grad_norm = gradients.norm(2, dim=1)
                        gradient_penalty = 10.0 * ((grad_norm - 1) ** 2).mean()
                    else:
                        print("Warning: The gradient is None, using a zero gradient penalty")
                        gradient_penalty = torch.tensor(0.0).to(video_feat.device)
                else:
                    print("Warning: Gradient is not required for input or output, skip gradient penalty calculation")
                    gradient_penalty = torch.tensor(0.0).to(video_feat.device)
                
            except Exception as e:
                print(f"The gradient penalizes errors in computation: {e}")
                gradient_penalty = torch.tensor(0.0).to(video_feat.device)
            
            try:
                video_features = self.D[:-1](video_feat)
                aligned_features = self.D[:-1](aligned_audio)
                feature_matching_loss = F.mse_loss(aligned_features, video_features)
            except Exception as e:
                print(f"err: {e}")
                feature_matching_loss = torch.tensor(0.0).to(video_feat.device)
            
   
            adv_loss = d_loss + g_loss + gradient_penalty + self.lambda_fm * feature_matching_loss
        else:
            adv_loss = torch.tensor(0.0).to(video_feat.device)
        
        return aligned_audio, adv_loss, gamma

class ImplicitAlign(nn.Module):
    def __init__(self, hidden_dim, max_iter=3, temperature=0.07):
        super().__init__()
        self.max_iter = max_iter
        self.hidden_dim = hidden_dim
        
        self.alpha = nn.Parameter(torch.zeros(hidden_dim))
        
        self.similarity_net = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.W = nn.Sequential(
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.LayerNorm(4*hidden_dim),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
        
        self.grad_norm = nn.LayerNorm(hidden_dim)
        
        self.register_buffer('gamma', torch.tensor([0.001, 0.01, 0.1, 1, 10, 100]))
        
    def gaussian_kernel_matrix(self, x, y, gamma):
        """
        Args:
            x: [B*T1, D] group 1
            y: [B*T2, D] group 2 
            gamma
        Returns:
            kernel_matrix: [B*T1, B*T2] 
        """
        x_norm = torch.sum(x**2, dim=1, keepdim=True)  # [B*T1, 1]
        y_norm = torch.sum(y**2, dim=1, keepdim=True)  # [B*T2, 1]
        
        dist_sq = x_norm + y_norm.t() - 2 * torch.mm(x, y.t())  # [B*T1, B*T2]
        
        dist_sq = torch.clamp(dist_sq, min=0.0)
        
        return torch.exp(-gamma * dist_sq)
    
    def compute_mmd_loss_sampled(self, x, y, max_samples=1024):
        """
        Args:
            x: [B, T, D] group1
            y: [B, T, D] group2
            max_samples
        Returns:
            mmd_loss
        """
        B, T, D = x.shape
        
        # reshape
        x_flat = x.view(-1, D)  # [B*T, D]
        y_flat = y.view(-1, D)  # [B*T, D]
        
        n = x_flat.size(0)
        if n <= 1:
            return torch.tensor(0.0).to(x.device)
        
        if n > max_samples:
            indices = torch.randperm(n, device=x.device)[:max_samples]
            x_flat = x_flat[indices]
            y_flat = y_flat[indices]
            n = max_samples
        
        mmd_loss = 0.0
        
        for gamma in self.gamma:

            K_xx = self.gaussian_kernel_matrix(x_flat, x_flat, gamma)
            K_yy = self.gaussian_kernel_matrix(y_flat, y_flat, gamma)
            K_xy = self.gaussian_kernel_matrix(x_flat, y_flat, gamma)
            
 
            K_xx_sum = (K_xx.sum() - K_xx.diag().sum()) / (n * (n - 1))
            K_yy_sum = (K_yy.sum() - K_yy.diag().sum()) / (n * (n - 1))
            K_xy_sum = K_xy.sum() / (n * n)
            
            mmd_loss += K_xx_sum + K_yy_sum - 2 * K_xy_sum
        
        return torch.clamp(mmd_loss / len(self.gamma), min=0.0)
    
    def forward(self, query, value):
        if self.training:
            return self._forward_train(query, value)
        else:
            return self._forward_eval(query, value)
    
    def _forward_train(self, query, value):
        B, T, D = query.shape
        R = self.alpha[None, None, :].expand(B, T, -1).clone().requires_grad_(True)
        
        for _ in range(self.max_iter):
            with torch.enable_grad():
  
                sim_weight = self.similarity_net(torch.cat([query+R, value], -1))
                
  
                weighted_value = value * sim_weight
                F_val = self.W(torch.cat([query+R, weighted_value], -1)) - (query+R)
                
 
                grad = torch.autograd.grad(F_val.mean(), R, create_graph=True)[0]
            
            with torch.no_grad():
                # R update
                R += -self.grad_norm(grad) * 0.1
        

        aligned_query = query + R
        
        mmd_loss = self.compute_mmd_loss_sampled(aligned_query, value, max_samples=1024)
        
        return aligned_query, mmd_loss
    
    def _forward_eval(self, query, value):

        sim_weight = self.similarity_net(torch.cat([query + self.alpha[None, None, :], value], -1))
        aligned_query = query + self.alpha[None, None, :] * sim_weight
        

        return aligned_query, torch.tensor(0.0).to(query.device)

class GradientReversal(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.alpha)

class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None