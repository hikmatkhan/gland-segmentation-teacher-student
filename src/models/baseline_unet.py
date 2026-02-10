#!/usr/bin/env python3
"""
Baseline UNet Implementation for Gland Segmentation
Simple, clean UNet architecture for baseline comparison with nnUNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any


class DoubleConv(nn.Module):
    """Double convolution block (conv -> BN -> ReLU -> conv -> BN -> ReLU)"""

    def __init__(self, in_channels: int, out_channels: int,
                 activation: str = 'relu', normalization: str = 'batch',
                 dropout: float = 0.0):
        super().__init__()

        # Activation function
        if activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Normalization
        if normalization == 'batch':
            norm1 = nn.BatchNorm2d(out_channels)
            norm2 = nn.BatchNorm2d(out_channels)
        elif normalization == 'group':
            norm1 = nn.GroupNorm(8, out_channels)
            norm2 = nn.GroupNorm(8, out_channels)
        elif normalization == 'instance':
            norm1 = nn.InstanceNorm2d(out_channels)
            norm2 = nn.InstanceNorm2d(out_channels)
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            norm1,
            act_fn
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        layers.extend([
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            norm2,
            act_fn
        ])

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, **kwargs)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, **kwargs):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After concatenation, we have in_channels + out_channels from skip connection
            self.conv = DoubleConv(in_channels + out_channels, out_channels, **kwargs)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            # After concatenation, we have in_channels // 2 + out_channels from skip connection
            self.conv = DoubleConv(in_channels, out_channels, **kwargs)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class BaselineUNet(nn.Module):
    """
    Simple UNet implementation for baseline comparison

    Args:
        input_channels: Number of input channels (default: 3 for RGB)
        num_classes: Number of output classes (default: 4 for Background, Benign, Malignant, PDC)
        depth: Network depth (number of downsampling levels)
        initial_channels: Starting channel count
        channel_multiplier: Factor to multiply channels at each level
        activation: Activation function ('relu', 'leaky_relu', 'gelu')
        normalization: Normalization type ('batch', 'group', 'instance')
        dropout: Dropout probability
        bilinear: Use bilinear upsampling instead of transpose conv
        enable_hooks: Enable feature hooks for classification
    """

    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 4,
                 depth: int = 4,
                 initial_channels: int = 64,
                 channel_multiplier: int = 2,
                 activation: str = 'relu',
                 normalization: str = 'batch',
                 dropout: float = 0.1,
                 bilinear: bool = True,
                 enable_hooks: bool = True):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.depth = depth
        self.enable_hooks = enable_hooks

        # Store architecture parameters
        self.config = {
            'input_channels': input_channels,
            'num_classes': num_classes,
            'depth': depth,
            'initial_channels': initial_channels,
            'channel_multiplier': channel_multiplier,
            'activation': activation,
            'normalization': normalization,
            'dropout': dropout,
            'bilinear': bilinear
        }

        # Calculate channel sizes for each level
        channels = [initial_channels * (channel_multiplier ** i) for i in range(depth + 1)]

        # Common conv kwargs
        conv_kwargs = {
            'activation': activation,
            'normalization': normalization,
            'dropout': dropout
        }

        # Initial convolution
        self.inc = DoubleConv(input_channels, channels[0], **conv_kwargs)

        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        for i in range(depth):
            self.encoder.append(Down(channels[i], channels[i + 1], **conv_kwargs))

        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        for i in range(depth):
            in_ch = channels[depth - i]      # Current level channels
            out_ch = channels[depth - i - 1]  # Output channels (skip connection level)
            self.decoder.append(Up(in_ch, out_ch, bilinear, **conv_kwargs))

        # Output convolution
        self.outc = OutConv(channels[0], num_classes)

        # Feature hooks for classification
        self.bottleneck_features = None
        if enable_hooks:
            self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture bottleneck features"""
        def hook_fn(module, input, output):
            # Store the deepest encoder features (bottleneck)
            self.bottleneck_features = output

        # Register hook on the last encoder stage (bottleneck)
        if len(self.encoder) > 0:
            self.encoder[-1].register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Reset bottleneck features
        self.bottleneck_features = None

        # Initial convolution
        x = self.inc(x)

        # Store skip connections
        skip_connections = [x]

        # Encoder path
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)

        # Remove the last skip connection (it's the bottleneck)
        skip_connections = skip_connections[:-1]

        # Decoder path
        for i, up in enumerate(self.decoder):
            skip = skip_connections[-(i + 1)]
            x = up(x, skip)

        # Output
        logits = self.outc(x)

        return logits

    def get_bottleneck_channels(self) -> int:
        """Get the number of channels in the bottleneck layer"""
        channels = [self.config['initial_channels'] * (self.config['channel_multiplier'] ** i)
                   for i in range(self.config['depth'] + 1)]
        return channels[-1]  # Last element is bottleneck

    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get feature maps at different levels for visualization/analysis

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Dictionary of feature maps at different levels
        """
        features = {}

        # Initial convolution
        x = self.inc(x)
        features['level_0'] = x

        # Encoder path
        for i, down in enumerate(self.encoder):
            x = down(x)
            features[f'level_{i + 1}'] = x

        features['bottleneck'] = x
        return features


def create_baseline_unet(input_channels: int = 3,
                        num_classes: int = 4,
                        depth: int = 4,
                        initial_channels: int = 64,
                        **kwargs) -> BaselineUNet:
    """
    Factory function to create baseline UNet

    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        depth: Network depth
        initial_channels: Starting channel count
        **kwargs: Additional arguments passed to BaselineUNet

    Returns:
        BaselineUNet instance
    """
    return BaselineUNet(
        input_channels=input_channels,
        num_classes=num_classes,
        depth=depth,
        initial_channels=initial_channels,
        **kwargs
    )


def test_baseline_unet():
    """Test function for baseline UNet"""
    print("🧪 Testing Baseline UNet...")

    # Test parameters
    batch_size = 2
    input_channels = 3
    num_classes = 4
    height, width = 256, 256

    # Create model
    model = create_baseline_unet(
        input_channels=input_channels,
        num_classes=num_classes,
        depth=4,
        initial_channels=64
    )

    # Test input
    x = torch.randn(batch_size, input_channels, height, width)

    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Bottleneck channels: {model.get_bottleneck_channels()}")

    # Forward pass
    with torch.no_grad():
        output = model(x)
        print(f"✅ Input shape: {x.shape}")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Bottleneck features captured: {model.bottleneck_features is not None}")
        if model.bottleneck_features is not None:
            print(f"✅ Bottleneck shape: {model.bottleneck_features.shape}")

        # Test feature maps
        feature_maps = model.get_feature_maps(x)
        print(f"✅ Feature maps levels: {list(feature_maps.keys())}")

    print("🎉 Baseline UNet test completed successfully!")
    return model


if __name__ == "__main__":
    test_baseline_unet()