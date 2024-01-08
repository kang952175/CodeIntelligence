import torch    
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

# 3)
class TimeEmbedding(nn.Module):
    
    def __init__(self, n_embed: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed) # 임베딩 4개
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # x : (1, 320)
        
        x = self.linear_1(x)
        
        x = F.silu(x)
        
        x = self.linear_2(x)
        
        # (1, 1280)
        return x

# 7)
class UNET_ResidualBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, n_time = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged =  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, feature, time):
        # feature: (b, In_channels, height, width)
        # time (1, 1280)
        
        residue = feature
        
        feature = self.groupnorm_feature(feature)
        
        feature = F.silu(feature)
        
        feature = self.conv_feature(feature)
        
        time = F.silu(time)
        
        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1) # time haven't b, c
        
        merged = self.groupnorm_merged(merged)
        
        merged = F.silu(merged)
        
        merged = self.conv_merged(merged)
        
        return merged + self.residual_layer(residue)
# 특정 프롬프트를 조건으로 사용하여 특정 시간 단계의 노이즈가 있는 이미지에서 노이즈를 감지하는 학습 진행

# 8)
class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embed: int, d_context = 768 ):
        super().__init__()
        channels = n_head * n_embed
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias = False)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
    def forward(self, x, context):
        # x: (b, f, h, w)
        # context : (b ,seq_len, dim)
        
        residue_long = x
        
        x = self.groupnorm(x)
        
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (b, f, h, w) -> (b, f, h * w)
        x = x.view((n, c, h * w))
        
        # (b, f, h * w) -> (b, h * w, f)
        x = x.transpose(-1, -2)
        
        # Normalization + self attention with skip connection
        
        residue_short = x
        
        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residue_short
        
        residue_short = x
        
        # Normalization + Cross Attention with skip connection
        x = self.layernorm_2(x)
        
        # Cross Attention
        self.attention_2(x, context)
        
        x += residue_short
        
        residue_short = x
        
        # Normalization + FF with GeGLU and skip connection
        
        x = self.layernorm_3(x)
        
        x, gate = self.linear_geglu_1(x).chunk(2, dim = -1)
        
        x = x * F.gelu(gate)
        
        x = self.linear_geglu_2(x)
        
        x += residue_short
        
        # (b, h * w, f) -> (b, f, h * w)
        x = x.transpose(-1, -2)
        
        x = x.view((n, c, h, w))
        
        return self.conv_output(x) + residue_long
        

# 5)
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)

# 2)
class SwitchSequential(nn.Sequential):
    
    def forward(self, x:torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context) # latent and prompt attention
                
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

# 4)
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (batch, 320, height / 8, width / 8) -> (batch, 320, height / 16, width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size = 3, stride = 2, padding = 1)),
            
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(640,  640), UNET_AttentionBlock(8, 80)),
            
            # (batch, 640, height / 16, width / 16) -> (batch, 640, height / 32, width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size = 3, stride = 2, padding = 1)),
            
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8,160)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8,160)),
            
            # (batch, 1280, height / 32, width / 32) -> (batch, 1280, height / 64, width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size = 3, stride = 2, padding = 1)),
            
            SwitchSequential(UNET_ResidualBlock(1280,  1280)),
            
            # (batch, 1280, height / 64, width / 64) -> (batch, 1280, height / 64, width / 64)
            SwitchSequential(UNET_ResidualBlock(1280,  1280)),
            
        ])
        
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            
            UNET_AttentionBlock(8, 160),
            
            UNET_ResidualBlock(1280, 1280),
        )
        
        self.decoders = nn.ModuleList([
            # (batch, 2560, height / 64, width/ 64) -> (batch, 1280, height / 64, width/ 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
        ])

# 6)
class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels : int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels,out_channels, kernel_size= 3, padding= 1)
        
    def forward(self, x):
        # x : (b, 320, h / 8, w / 8 )
        
        x = self.groupnorm(x)
        
        x = F.silu(x)
        
        x = self.conv(x)
        
        # (b, 4, h / 8, w / 8)
        return x

# 1)
class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
        
    def forward(self, latent : torch.Tensor, context : torch.Tensor, time:torch.Tensor):
        # latent : (batch_size, 4, height / 8, width / 8)
        # context: (batch_size, seq_len, dim)
        # time : (1, 320)
        
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time) # 노이즈 제거의 어느 단계 도달한지 정보
        
        # (batch, 3, height /8, width / 8 ) -> (batch, 320, height / 8 , width / 8)
        output = self.unet(latent, context, time)
        
        # (batch, 320, height / 8 , width / 8) -> (batch, 4, height / 8 , width / 8)
        output = self.final(output) 
        
        # 출력 차원은 입력 차원과 일치해야 반복적으로 노이즈 제거 과정 수행 가능
        return output 
