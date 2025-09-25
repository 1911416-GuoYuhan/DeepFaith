import torch.nn as nn
import torch

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, image_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        """
        Map input tensor to patch.
        Args:
            image_size: input image size
            patch_size: patch size
            in_c: number of input channels
            embed_dim: embedding dimension. dimension = patch_size * patch_size * in_c
            norm_layer: The function of normalization
        """
        super().__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # The input tensor is divided into patches using 16x16 convolution
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C] B*196*768
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchRecover(nn.Module):

    def __init__(self, original_size=224, patch_size=16, out_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.original_size = original_size
        self.patch_size = patch_size
        self.grid_size = original_size // patch_size
        
        self.proj = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=out_c,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, N, D = x.shape
        assert N == self.grid_size**2, \
            f"Input sequence length ({N}) doesn't match grid size ({self.grid_size}^2)."

        x = self.norm(x)

        x = x.permute(0, 2, 1).view(B, D, self.grid_size, self.grid_size)
        
        x = self.proj(x)
        return x

# 测试用例
if __name__ == '__main__':
    # 正向变换
    embedder = PatchEmbed()
    dummy_input = torch.randn(2, 3, 224, 224)
    patches = embedder(dummy_input)  
    
    recover = PatchRecover()
    output = recover(patches) 
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Recovered shape: {output.shape}")
