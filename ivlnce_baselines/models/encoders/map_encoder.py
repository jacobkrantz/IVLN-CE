from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBRA(nn.Module):
    """Conv -> BatchNorm -> ReLU -> AvgPool"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=7, padding="same"
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        return self.conv(x)


class SemanticMapEncoder(nn.Module):
    """
    Jointly encodes semantic and occupancy maps to a vector.
    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the returned embedding vector
        num_semantic_classes: # of semantic classes to expect from the input
    """

    def __init__(
        self,
        observation_space,
        num_semantic_classes: int = 13,
        ch: int = 32,
        last_ch_mult: int = 8,
        trainable: bool = True,
        from_pretrained: bool = False,
        checkpoint: Optional[str] = None,
    ):
        super().__init__()
        for k in ["occupancy_map", "semantic_map"]:
            if k not in observation_space.spaces:
                raise ValueError(f"key `{k}` expected in observation space.")

        self._map_dimensions = observation_space.spaces["occupancy_map"].shape
        self._num_semantic_classes = num_semantic_classes
        self.last_ch_mult = last_ch_mult

        self._ch = ch
        self.cnn = nn.Sequential(
            CBRA(14, ch),
            CBRA(ch, ch * 2),
            CBRA(ch * 2, ch * 4),
            CBRA(ch * 4, ch * last_ch_mult),
        )

        if from_pretrained:
            ckpt = torch.load(checkpoint, map_location="cpu")["state_dict"]
            prefix = "encoder.cnn."
            state_dict = {
                k[len(prefix) :]: v
                for k, v in ckpt.items()
                if k.startswith(prefix)
            }
            self.cnn.load_state_dict(state_dict)

        for param in self.cnn.parameters():
            param.requires_grad_(trainable)

        if not trainable:
            self.eval()

    @property
    def output_shape(self):
        nrows = self._map_dimensions[0]
        ncols = self._map_dimensions[1]
        div = 2 ** 4  # there are 4 instances of 1/2 avg pool
        return (self._ch * self.last_ch_mult, nrows // div, ncols // div)

    def generate_map_features(self, observations):
        occupancy = observations["occupancy_map"].unsqueeze(1)
        semantic = observations["semantic_map"].long()
        semantic = F.one_hot(semantic, self._num_semantic_classes)
        semantic = semantic.permute(0, 3, 1, 2)
        return torch.cat((occupancy, semantic), 1).to(dtype=torch.float)

    def forward(self, observations):
        for k in ["occupancy_map", "semantic_map"]:
            if k not in observations:
                raise ValueError(f"Observation `{k}` is missing.")

        return self.cnn(self.generate_map_features(observations))
