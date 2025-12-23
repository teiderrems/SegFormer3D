"""
SegFormer3D : Architecture de Segmentation 3D basée sur Transformateur

Ce module implémente SegFormer3D, une adaptation 3D du modèle SegFormer pour la segmentation
sémantique d'images médicales volumétriques. L'architecture combine un encodeur Transformer
hiérarchique avec un décodeur de fusion multi-échelle.

Architecture:
    Encodeur: MixVisionTransformer (4 étapes pyramidales)
    Décodeur: SegFormerDecoderHead (fusion multi-échelle avec MLP linéaires)

Caractéristiques principales:
    - Attention réduite spatialement pour efficacité
    - Opérations 3D pour traiter des volumes complets
    - Initialisation des poids personnalisée (Kaiming/Truncated Normal)
    - Support du calcul distribué (DDP)

Références académiques:
    - SegFormer: "Simple and Efficient Design for Semantic Segmentation with Transformers"
    - Adapté pour les images 3D (volumes médicaux)

Exemple d'utilisation:
    config = {
        "model_parameters": {
            "in_channels": 4,
            "embed_dims": [32, 64, 160, 256],
            "num_classes": 3,
            ...
        }
    }
    model = build_segformer3d_model(config)
    output = model(input_volume)  # (B, 4, D, H, W) -> (B, 3, D, H, W)
"""

import torch
import math
import copy
from torch import nn
from einops import rearrange
from functools import partial
from typing import Tuple, List

def build_segformer3d_model(config=None):
    """Crée une instance du modèle SegFormer3D à partir d'un dictionnaire de configuration.
    
    Args:
        config (dict): Dictionnaire de configuration contenant:
            - config["model_parameters"]["in_channels"]: Nombre de canaux d'entrée (généralement 4 pour T1, T1CE, T2, FLAIR)
            - config["model_parameters"]["sr_ratios"]: Taux de réduction spatiale pour l'attention
            - config["model_parameters"]["embed_dims"]: Dimensions de plongement à chaque étape
            - config["model_parameters"]["patch_kernel_size"]: Taille du noyau pour plongement de patchs
            - config["model_parameters"]["patch_stride"]: Pas de convolution
            - config["model_parameters"]["patch_padding"]: Rembourrage
            - config["model_parameters"]["mlp_ratios"]: Ratio d'expansion du MLP
            - config["model_parameters"]["num_heads"]: Nombre de têtes d'attention
            - config["model_parameters"]["depths"]: Nombre de blocs Transformer
            - config["model_parameters"]["decoder_head_embedding_dim"]: Dimension de la tête du décodeur
            - config["model_parameters"]["num_classes"]: Nombre de classes de segmentation
            - config["model_parameters"]["decoder_dropout"]: Taux de dropout du décodeur
    
    Returns:
        SegFormer3D: Instance du modèle configuré
    
    Exemple:
        >>> config = load_config("config.yaml")
        >>> model = build_segformer3d_model(config)
    """
    model=SegFormer3D(
        in_channels=config["model_parameters"]["in_channels"],
        sr_ratios=config["model_parameters"]["sr_ratios"],
        embed_dims=config["model_parameters"]["embed_dims"],
        patch_kernel_size=config["model_parameters"]["patch_kernel_size"],
        patch_stride=config["model_parameters"]["patch_stride"],
        patch_padding=config["model_parameters"]["patch_padding"],
        mlp_ratios=config["model_parameters"]["mlp_ratios"],
        num_heads=config["model_parameters"]["num_heads"],
        depths=config["model_parameters"]["depths"],
        decoder_head_embedding_dim=config["model_parameters"][
            "decoder_head_embedding_dim"
        ],
        num_classes=config["model_parameters"]["num_classes"],
        decoder_dropout=config["model_parameters"]["decoder_dropout"],
    )
    return model


class SegFormer3D(nn.Module):
    """Modèle SegFormer3D complet : Encodeur + Décodeur pour segmentation 3D.
    
    Architecture:
        1. Encodeur MixVisionTransformer: Extrait 4 niveaux pyramidaux de caractéristiques
        2. Décodeur SegFormerDecoderHead: Fusionne et upsamples les caractéristiques
    
    Processus en avant:
        Input(B, C_in, D, H, W) 
            -> Encodeur (4 étapes) -> [c1, c2, c3, c4]
            -> Décodeur (fusion multi-échelle, upsampling x4)
            -> Output(B, num_classes, D, H, W)
    
    Avantages:
        - Efficace en mémoire grâce à l'attention réduite spatialement
        - Opérations 3D natives pour volumes médicaux
        - Fusion multi-échelle robuste
        - Initialisation des poids optimisée
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        sr_ratios: list = [4, 2, 1, 1],
        embed_dims: list = [32, 64, 160, 256],
        patch_kernel_size: list = [7, 3, 3, 3],
        patch_stride: list = [4, 2, 2, 2],
        patch_padding: list = [3, 1, 1, 1],
        mlp_ratios: list = [4, 4, 4, 4],
        num_heads: list = [1, 2, 5, 8],
        depths: list = [2, 2, 2, 2],
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 3,
        decoder_dropout: float = 0.0,
    ):
        """Initialise le modèle SegFormer3D.
        
        Args:
            in_channels (int): Nombre de canaux d'entrée (ex: 4 pour BraTS)
            sr_ratios (list): Taux de réduction spatiale pour l'attention à chaque étape.
                Exemple: [4, 2, 1, 1] réduit progressivement les clés/valeurs
            embed_dims (list): Dimensions de plongement à chaque étape.
                Exemple: [32, 64, 160, 256] pour progression graduelle
            patch_kernel_size (list): Taille du noyau de convolution pour plongement de patchs
            patch_stride (list): Pas de convolution (détermine la réduction spatiale)
            patch_padding (list): Rembourrage de convolution
            mlp_ratios (list): Ratio d'expansion du MLP à chaque étape.
                Exemple: [4, 4, 4, 4] = dimension MLP = 4 * embed_dim
            num_heads (list): Nombre de têtes d'attention par étape
            depths (list): Nombre de blocs Transformer par étape
            decoder_head_embedding_dim (int): Dimension de plongement de la tête du décodeur (256)
            num_classes (int): Nombre de classes de segmentation (ex: 3 pour BraTS)
            decoder_dropout (float): Taux de dropout après fusion dans le décodeur
        """
        super().__init__()
        self.segformer_encoder = MixVisionTransformer(
            in_channels=in_channels,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mlp_ratios=mlp_ratios,
            num_heads=num_heads,
            depths=depths,
        )
        # decoder takes in the feature maps in the reversed order
        reversed_embed_dims = embed_dims[::-1]
        self.segformer_decoder = SegFormerDecoderHead(
            input_feature_dims=reversed_embed_dims,
            decoder_head_embedding_dim=decoder_head_embedding_dim,
            num_classes=num_classes,
            dropout=decoder_dropout,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialise les poids du modèle selon le type de couche.
        
        Stratégies d'initialisation utilisées:
            - Linear: Normale tronquée (std=0.02)
            - LayerNorm: Poids=1, Bias=0
            - BatchNorm: Poids=1, Bias=0
            - Conv2d/Conv3d: Kaiming normal (fan_out)
        
        Args:
            m (nn.Module): Module à initialiser
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe en avant du modèle complet.
        
        Processus:
            1. Encoder l'entrée en caractéristiques pyramidales (4 niveaux)
            2. Décoder et fusionner les caractéristiques
            3. Générer la carte de segmentation finale
        
        Args:
            x (torch.Tensor): Volume d'entrée (B, in_channels, D, H, W)
        
        Returns:
            torch.Tensor: Prédictions segmentation (B, num_classes, D, H, W)
        
        Exemple:
            >>> model = SegFormer3D(in_channels=4, num_classes=3)
            >>> input_vol = torch.randn(1, 4, 128, 128, 128)
            >>> output = model(input_vol)  # (1, 3, 128, 128, 128)
        """
        # Encodage en caractéristiques pyramidales
        x = self.segformer_encoder(x)
        
        # Déballage des 4 niveaux de caractéristiques
        # c1: résolution maximale (1/4), c4: résolution minimale
        c1, c2, c3, c4 = x[0], x[1], x[2], x[3]
        
        # Décodage avec fusion multi-échelle
        x = self.segformer_decoder(c1, c2, c3, c4)
        return x
    
# ----------------------------------------------------- ENCODEUR 3D AVEC PATCHS PYRAMIDAUX -----

class PatchEmbedding(nn.Module):
    """Couche de plongement de patchs pour volumes 3D.
    
    Convertit un volume d'entrée en patchs intégrés via convolution 3D, suivi de normalisation.
    
    Processus:
        Input (B, C_in, D, H, W)
            -> Conv3d (C_in -> embed_dim)
            -> Aplatissement et transposition
            -> LayerNorm
            -> Output (B, num_patches, embed_dim)
    
    où num_patches = (D/stride) * (H/stride) * (W/stride)
    """
    
    def __init__(
        self,
        in_channel: int = 4,
        embed_dim: int = 768,
        kernel_size: int = 7,
        stride: int = 4,
        padding: int = 3,
    ):
        """Initialise le module de plongement de patchs.
        
        Args:
            in_channel (int): Nombre de canaux d'entrée (ex: 4 pour modalités MRI)
            embed_dim (int): Dimension du vecteur de plongement
            kernel_size (int): Taille du noyau de convolution 3D
            stride (int): Pas de la convolution (détermine la réduction spatiale)
            padding (int): Rembourrage de convolution
        """
        super().__init__()
        self.patch_embeddings = nn.Conv3d(
            in_channel,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """Crée les patchs intégrés à partir du volume d'entrée.
        
        Args:
            x (torch.Tensor): Volume d'entrée (B, in_channel, D, H, W)
        
        Returns:
            torch.Tensor: Patchs intégrés (B, num_patches, embed_dim)
        """
        # Applique la convolution 3D pour créer les patchs
        patches = self.patch_embeddings(x)
        # Transforme de (B, embed_dim, D', H', W') en (B, D'*H'*W', embed_dim)
        patches = patches.flatten(2).transpose(1, 2)
        # Normalise par couche
        patches = self.norm(patches)
        return patches


class SelfAttention(nn.Module):
    """Mécanisme d'attention multi-tête avec réduction spatiale (Spatial Reduction Attention).
    
    Implémente l'attention dans le contexte SegFormer:
    - Réduit les clés et valeurs par sr_ratio pour efficacité mémoire
    - Utilise scaled_dot_product_attention optimisé (PyTorch 2.0+)
    - Support complet du dropout pour régularisation
    
    Complexité mémoire:
        Sans réduction: O(N²) où N = nombre de patchs
        Avec réduction: O(N²/r) où r = sr_ratio
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        """Initialise le module d'attention multi-tête.
        
        Args:
            embed_dim (int): Dimension d'intégration (dimension cachée)
            num_heads (int): Nombre de têtes d'attention
            sr_ratio (int): Taux de réduction spatiale pour clés/valeurs.
                            sr_ratio=1 = pas de réduction, sr_ratio=4 = réduction 4x
            qkv_bias (bool): Si True, ajoute un biais aux projections linéaires
            attn_dropout (float): Taux de dropout appliqué aux poids d'attention
            proj_dropout (float): Taux de dropout appliqué à la projection finale
        
        Raises:
            AssertionError: Si embed_dim n'est pas divisible par num_heads
        """
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dim should be divisible by number of heads!"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        # embedding dimesion of each attention head
        self.attention_head_dim = embed_dim // num_heads
        self.scale = self.attention_head_dim ** -0.5

        # The same input is used to generate the query, key, and value,
        # (batch_size, num_patches, hidden_size) -> (batch_size, num_patches, attention_head_size)
        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.attn_dropout_p = attn_dropout
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applique l'attention multi-tête avec réduction spatiale.
        
        Processus:
            1. Projette l'entrée en Queries (Q)
            2. Si sr_ratio > 1: Réduit spatialement pour Keys (K) et Values (V)
            3. Calcule Attention = softmax(Q·K^T/√d)·V
            4. Projette et applique dropout
        
        Args:
            x (torch.Tensor): Patchs intégrés (B, N, embed_dim)
        
        Returns:
            torch.Tensor: Sortie attention (B, N, embed_dim)
        
        Mathématiques:
            Attention(Q,K,V) = softmax(Q @ K^T / √(d_k)) @ V
        """
        B, N, C = x.shape

        # (batch_size, num_head, sequence_length, embed_dim)
        q = (
            self.query(x)
            .reshape(B, N, self.num_heads, self.attention_head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        if self.sr_ratio > 1:
            n = cube_root(N)
            # (batch_size, sequence_length, embed_dim) -> (batch_size, embed_dim, patch_D, patch_H, patch_W)
            x_ = x.permute(0, 2, 1).reshape(B, C, n, n, n).contiguous()
            # (batch_size, embed_dim, patch_D, patch_H, patch_W) -> (batch_size, embed_dim, patch_D/sr_ratio, patch_H/sr_ratio, patch_W/sr_ratio)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            # (batch_size, embed_dim, patch_D/sr_ratio, patch_H/sr_ratio, patch_W/sr_ratio) -> (batch_size, sequence_length, embed_dim)
            # normalizing the layer
            x_ = self.sr_norm(x_)
            # (batch_size, num_patches, hidden_size)
            kv = (
                self.key_value(x_)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            # (2, batch_size, num_heads, num_sequence, attention_head_dim)
        else:
            # (batch_size, num_patches, hidden_size)
            kv = (
                self.key_value(x)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            # (2, batch_size, num_heads, num_sequence, attention_head_dim)

        k, v = kv[0].contiguous(), kv[1].contiguous()

        # Use PyTorch's optimized scaled_dot_product_attention when available (PyTorch 2.0+)
        # Falls back to manual computation for older versions
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=False
            )
        else:
            attention_score = (q @ k.transpose(-2, -1)) * self.scale
            attnention_prob = attention_score.softmax(dim=-1)
            if self.training and self.attn_dropout_p > 0:
                attnention_prob = torch.nn.functional.dropout(attnention_prob, p=self.attn_dropout_p, training=True)
            out = attnention_prob @ v
        
        out = out.transpose(1, 2).reshape(B, N, C).contiguous()
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: int = 2,
        num_heads: int = 8,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        """
        embed_dim : hidden size of the PatchEmbedded input
        mlp_ratio: at which rate increasse the projection dim of the embedded patch in the _MLP component
        num_heads: number of attention heads
        sr_ratio: the rate at which to down sample the sequence length of the embedded patch
        qkv_bias: whether or not the linear projection has bias
        attn_dropout: the dropout rate of the attention component
        proj_dropout: the dropout rate of the final linear projection
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _MLP(in_feature=embed_dim, mlp_ratio=mlp_ratio, dropout=0.0)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        sr_ratios: list = [8, 4, 2, 1],
        embed_dims: list = [64, 128, 320, 512],
        patch_kernel_size: list = [7, 3, 3, 3],
        patch_stride: list = [4, 2, 2, 2],
        patch_padding: list = [3, 1, 1, 1],
        mlp_ratios: list = [2, 2, 2, 2],
        num_heads: list = [1, 2, 5, 8],
        depths: list = [2, 2, 2, 2],
    ):
        """
        in_channels: number of the input channels
        img_volume_dim: spatial resolution of the image volume (Depth, Width, Height)
        sr_ratios: the rates at which to down sample the sequence length of the embedded patch
        embed_dims: hidden size of the PatchEmbedded input
        patch_kernel_size: kernel size for the convolution in the patch embedding module
        patch_stride: stride for the convolution in the patch embedding module
        patch_padding: padding for the convolution in the patch embedding module
        mlp_ratio: at which rate increasse the projection dim of the hidden_state in the mlp
        num_heads: number of attenion heads
        depth: number of attention layers
        """
        super().__init__()

        # patch embedding at different Pyramid level
        self.embed_1 = PatchEmbedding(
            in_channel=in_channels,
            embed_dim=embed_dims[0],
            kernel_size=patch_kernel_size[0],
            stride=patch_stride[0],
            padding=patch_padding[0],
        )
        self.embed_2 = PatchEmbedding(
            in_channel=embed_dims[0],
            embed_dim=embed_dims[1],
            kernel_size=patch_kernel_size[1],
            stride=patch_stride[1],
            padding=patch_padding[1],
        )
        self.embed_3 = PatchEmbedding(
            in_channel=embed_dims[1],
            embed_dim=embed_dims[2],
            kernel_size=patch_kernel_size[2],
            stride=patch_stride[2],
            padding=patch_padding[2],
        )
        self.embed_4 = PatchEmbedding(
            in_channel=embed_dims[2],
            embed_dim=embed_dims[3],
            kernel_size=patch_kernel_size[3],
            stride=patch_stride[3],
            padding=patch_padding[3],
        )

        # block 1
        self.tf_block1 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    sr_ratio=sr_ratios[0],
                    qkv_bias=True,
                )
                for _ in range(depths[0])
            ]
        )
        self.norm1 = nn.LayerNorm(embed_dims[0])

        # block 2
        self.tf_block2 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    sr_ratio=sr_ratios[1],
                    qkv_bias=True,
                )
                for _ in range(depths[1])
            ]
        )
        self.norm2 = nn.LayerNorm(embed_dims[1])

        # block 3
        self.tf_block3 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    sr_ratio=sr_ratios[2],
                    qkv_bias=True,
                )
                for _ in range(depths[2])
            ]
        )
        self.norm3 = nn.LayerNorm(embed_dims[2])

        # block 4
        self.tf_block4 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    sr_ratio=sr_ratios[3],
                    qkv_bias=True,
                )
                for _ in range(depths[3])
            ]
        )
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        out = []
        # at each stage these are the following mappings:
        # (batch_size, num_patches, hidden_state)
        # (num_patches,) -> (D, H, W)
        # (batch_size, num_patches, hidden_state) -> (batch_size, hidden_state, D, H, W)

        # stage 1
        x = self.embed_1(x)
        B, N, C = x.shape
        n = cube_root(N)
        for i, blk in enumerate(self.tf_block1):
            x = blk(x)
        x = self.norm1(x)
        # (B, N, C) -> (B, D, H, W, C) -> (B, C, D, H, W)
        x = x.reshape(B, n, n, n, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        # stage 2
        x = self.embed_2(x)
        B, N, C = x.shape
        n = cube_root(N)
        for i, blk in enumerate(self.tf_block2):
            x = blk(x)
        x = self.norm2(x)
        # (B, N, C) -> (B, D, H, W, C) -> (B, C, D, H, W)
        x = x.reshape(B, n, n, n, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        # stage 3
        x = self.embed_3(x)
        B, N, C = x.shape
        n = cube_root(N)
        for i, blk in enumerate(self.tf_block3):
            x = blk(x)
        x = self.norm3(x)
        # (B, N, C) -> (B, D, H, W, C) -> (B, C, D, H, W)
        x = x.reshape(B, n, n, n, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        # stage 4
        x = self.embed_4(x)
        B, N, C = x.shape
        n = cube_root(N)
        for i, blk in enumerate(self.tf_block4):
            x = blk(x)
        x = self.norm4(x)
        # (B, N, C) -> (B, D, H, W, C) -> (B, C, D, H, W)
        x = x.reshape(B, n, n, n, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        return out


class _MLP(nn.Module):
    def __init__(self, in_feature: int, mlp_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        out_feature = mlp_ratio * in_feature
        self.fc1 = nn.Linear(in_feature, out_feature)
        self.dwconv = DWConv(dim=out_feature)
        self.fc2 = nn.Linear(out_feature, in_feature)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        # added batchnorm (remove it ?)
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # (batch, patch_cube, hidden_size) -> (batch, hidden_size, D, H, W)
        # assuming D = H = W, i.e. cube root of the patch is an integer number!
        n = cube_root(N)
        x = x.transpose(1, 2).reshape(B, C, n, n, n).contiguous()
        x = self.dwconv(x)
        # added batchnorm (remove it ?)
        x = self.bn(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

###################################################################################
@torch.jit.script
def cube_root(n: int) -> int:
    """Compute integer cube root efficiently."""
    return int(round(n ** (1.0 / 3.0)))
    

###################################################################################
# ----------------------------------------------------- decoder -------------------
class MLP_(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.bn = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        # added batchnorm (remove it ?)
        x = self.bn(x)
        return x


###################################################################################
class SegFormerDecoderHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(
        self,
        input_feature_dims: list = [512, 320, 128, 64],
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.0,
    ):
        """
        input_feature_dims: list of the output features channels generated by the transformer encoder
        decoder_head_embedding_dim: projection dimension of the mlp layer in the all-mlp-decoder module
        num_classes: number of the output channels
        dropout: dropout rate of the concatenated feature maps
        """
        super().__init__()
        self.linear_c4 = MLP_(
            input_dim=input_feature_dims[0],
            embed_dim=decoder_head_embedding_dim,
        )
        self.linear_c3 = MLP_(
            input_dim=input_feature_dims[1],
            embed_dim=decoder_head_embedding_dim,
        )
        self.linear_c2 = MLP_(
            input_dim=input_feature_dims[2],
            embed_dim=decoder_head_embedding_dim,
        )
        self.linear_c1 = MLP_(
            input_dim=input_feature_dims[3],
            embed_dim=decoder_head_embedding_dim,
        )
        # convolution module to combine feature maps generated by the mlps
        self.linear_fuse = nn.Sequential(
            nn.Conv3d(
                in_channels=4 * decoder_head_embedding_dim,
                out_channels=decoder_head_embedding_dim,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm3d(decoder_head_embedding_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

        # final linear projection layer
        self.linear_pred = nn.Conv3d(
            decoder_head_embedding_dim, num_classes, kernel_size=1
        )

        # segformer decoder generates the final decoded feature map size at 1/4 of the original input volume size
        self.upsample_volume = nn.Upsample(
            scale_factor=4.0, mode="trilinear", align_corners=False
        )

    def forward(self, c1, c2, c3, c4):
       ############## _MLP decoder on C1-C4 ###########
        n, _, _, _, _ = c4.shape

        _c4 = (
            self.linear_c4(c4)
            .permute(0, 2, 1)
            .reshape(n, -1, c4.shape[2], c4.shape[3], c4.shape[4])
            .contiguous()
        )
        _c4 = torch.nn.functional.interpolate(
            _c4,
            size=c1.size()[2:],
            mode="trilinear",
            align_corners=False,
        )

        _c3 = (
            self.linear_c3(c3)
            .permute(0, 2, 1)
            .reshape(n, -1, c3.shape[2], c3.shape[3], c3.shape[4])
            .contiguous()
        )
        _c3 = torch.nn.functional.interpolate(
            _c3,
            size=c1.size()[2:],
            mode="trilinear",
            align_corners=False,
        )

        _c2 = (
            self.linear_c2(c2)
            .permute(0, 2, 1)
            .reshape(n, -1, c2.shape[2], c2.shape[3], c2.shape[4])
            .contiguous()
        )
        _c2 = torch.nn.functional.interpolate(
            _c2,
            size=c1.size()[2:],
            mode="trilinear",
            align_corners=False,
        )

        _c1 = (
            self.linear_c1(c1)
            .permute(0, 2, 1)
            .reshape(n, -1, c1.shape[2], c1.shape[3], c1.shape[4])
            .contiguous()
        )

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = self.upsample_volume(x)
        return x

###################################################################################
if __name__ == "__main__":
    input = torch.randint(
        low=0,
        high=255,
        size=(1, 4, 128, 128, 128),
        dtype=torch.float,
    )
    input = input.to("cuda:0")
    segformer3D = SegFormer3D().to("cuda:0")
    output = segformer3D(input)
    print(output.shape)


###################################################################################
