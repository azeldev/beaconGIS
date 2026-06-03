"""Siamese U-Net for building damage detection. Inference only."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SiameseUNet(nn.Module):
    """Siamese U-Net for building damage classification.
    Trained on xView2/xBD with SeResNeXt-50 (32x4d) encoder."""

    def __init__(self, encoder_name='seresnext50_32x4d', num_classes=5, pretrained=False):
        super().__init__()

        self.num_classes = num_classes

        # Stage count: 5 for ResNet/SeResNeXt/EfficientNet, 4 for ConvNeXt/Swin.
        # Probe 5 first, fall back to 4. The 5-stage branch keeps the legacy
        # dec4/dec3/dec2/dec1 attribute names so SeResNeXt-50 checkpoints
        # (M1, M2, loc_best.pth) load with strict=True.
        try:
            self.encoder = timm.create_model(
                encoder_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3, 4)
            )
        except (IndexError, RuntimeError):
            self.encoder = timm.create_model(
                encoder_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3)
            )

        # Batch size 2 (not 1): some encoders raise on BN with one sample.
        with torch.no_grad():
            dummy = torch.randn(2, 3, 128, 128)
            features = self.encoder(dummy)
            self.enc_channels = [f.shape[1] for f in features]
            shallowest_stride = 128 // features[0].shape[-1]

        n = len(self.enc_channels)
        self.n_stages = n

        # Feature fusion: [pre | post | |post-pre|] -> 1x channels
        self.fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch * 3, ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ) for ch in self.enc_channels
        ])

        if n == 5:
            # Attribute names MUST match training code for strict=True load.
            dec_channels = [128, 64, 32, 16]
            self.dec4 = DecoderBlock(self.enc_channels[4], self.enc_channels[3], dec_channels[0])
            self.dec3 = DecoderBlock(dec_channels[0], self.enc_channels[2], dec_channels[1])
            self.dec2 = DecoderBlock(dec_channels[1], self.enc_channels[1], dec_channels[2])
            self.dec1 = DecoderBlock(dec_channels[2], self.enc_channels[0], dec_channels[3])
            final_in = dec_channels[3]
        else:
            dec_channels = [128, 64, 32]
            self.dec_blocks = nn.ModuleList()
            in_ch = self.enc_channels[-1]
            for i in range(n - 1):
                skip_ch = self.enc_channels[-(i + 2)]
                self.dec_blocks.append(
                    DecoderBlock(in_ch, skip_ch, dec_channels[i])
                )
                in_ch = dec_channels[i]
            final_in = in_ch

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=shallowest_stride, mode='bilinear',
                        align_corners=False),
            nn.Conv2d(final_in, final_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(final_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_in, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[2:]

        pre = x[:, :3]
        post = x[:, 3:]

        pre_features = self.encoder(pre)
        post_features = self.encoder(post)

        fused_features = []
        for i, (pre_f, post_f) in enumerate(zip(pre_features, post_features)):
            diff_f = torch.abs(post_f - pre_f)
            combined_f = torch.cat([pre_f, post_f, diff_f], dim=1)
            fused = self.fusion[i](combined_f)
            fused_features.append(fused)

        if self.n_stages == 5:
            x = self.dec4(fused_features[4], fused_features[3])
            x = self.dec3(x, fused_features[2])
            x = self.dec2(x, fused_features[1])
            x = self.dec1(x, fused_features[0])
        else:
            x = fused_features[-1]
            for i, block in enumerate(self.dec_blocks):
                x = block(x, fused_features[-(i + 2)])

        x = self.final(x)

        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x


class LocalizationUNet(nn.Module):
    """Pre-image-only U-Net for binary building localization.

    Pairs with SiameseUNet during inference: the localization mask decides
    where buildings are, the classification net decides what damage class."""

    def __init__(self, encoder_name='seresnext50_32x4d', pretrained=False):
        super().__init__()

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)
        )

        with torch.no_grad():
            dummy = torch.randn(1, 3, 64, 64)
            features = self.encoder(dummy)
            self.enc_channels = [f.shape[1] for f in features]

        # Attribute names (dec*, final) MUST match training code for strict load.
        dec_channels = [128, 64, 32, 16]
        self.dec4 = DecoderBlock(self.enc_channels[4], self.enc_channels[3], dec_channels[0])
        self.dec3 = DecoderBlock(dec_channels[0], self.enc_channels[2], dec_channels[1])
        self.dec2 = DecoderBlock(dec_channels[1], self.enc_channels[1], dec_channels[2])
        self.dec1 = DecoderBlock(dec_channels[2], self.enc_channels[0], dec_channels[3])

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dec_channels[3], dec_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_channels[3], 1, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.encoder(x)

        y = self.dec4(features[4], features[3])
        y = self.dec3(y, features[2])
        y = self.dec2(y, features[1])
        y = self.dec1(y, features[0])
        y = self.final(y)

        if y.shape[2:] != input_size:
            y = F.interpolate(y, size=input_size, mode='bilinear', align_corners=False)

        return y
