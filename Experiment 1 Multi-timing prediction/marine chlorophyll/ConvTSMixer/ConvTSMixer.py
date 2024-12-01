#
#单层卷积效果较好

# import torch
# import torch.nn as nn
#
#
# class ResBlock(nn.Module):
#     def __init__(self, configs):
#         super(ResBlock, self).__init__()
#
#         self.temporal = nn.Sequential(
#             nn.Linear(configs.seq_len, configs.d_model),
#             nn.ReLU(),
#             nn.Linear(configs.d_model, configs.seq_len),
#             nn.Dropout(configs.dropout)
#         )
#
#         self.channel = nn.Sequential(
#             nn.Linear(configs.enc_in, configs.d_model),
#             nn.ReLU(),
#             nn.Linear(configs.d_model, configs.enc_in),
#             nn.Dropout(configs.dropout)
#         )
#
#     def forward(self, x):
#         # x: [B, L, D]
#         x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
#         x = x + self.channel(x)
#         return x
#
#
# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.layer = configs.e_layers
#
#         # Adding a one-dimensional convolutional layer for time series data
#         self.conv = nn.Conv1d(in_channels=configs.enc_in, out_channels=configs.enc_in, kernel_size=3, padding=1)
#         self.activation = nn.ReLU()
#
#         self.model = nn.ModuleList([ResBlock(configs) for _ in range(self.layer)])
#         self.pred_len = configs.pred_len
#         self.projection = nn.Linear(configs.seq_len, configs.pred_len)
#
#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         # Apply convolution and activation
#         x_enc = x_enc.transpose(1, 2)  # Reorder dimensions to [B, D, L] for convolution
#         x_enc = self.conv(x_enc)  # Apply convolution along the length of the time series
#         x_enc = self.activation(x_enc)
#         x_enc = x_enc.transpose(1, 2)  # Reorder back to [B, L, D] for further processing
#
#         # Process with ResBlocks
#         for i in range(self.layer):
#             x_enc = self.model[i](x_enc)
#         enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
#
#         return enc_out
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#         return dec_out[:, -self.pred_len:, :]  # Select the output for the prediction length
#
#
# class Config:
#     def __init__(self):
#         self.seq_len = 12
#         self.pred_len = 1
#         self.e_layers = 2
#         self.enc_in = 10
#         self.d_model = 32
#         self.dropout = 0.1
#         self.conv_channels = 16
#         self.conv_kernel_size = 3
#         self.conv_stride = 1
#         self.conv_padding = 1
#
#
# configs = Config()
# model = Model(configs)
#
# # Example tensor for demonstration
# example_input = torch.randn(10, configs.seq_len, configs.enc_in)
# example_output = model(example_input, None, None, None)
# print(example_output.shape)  # Output shape
#
#


# 三层卷积效果一般
# #
# import torch
# import torch.nn as nn
#
#
# class ResBlock(nn.Module):
#     def __init__(self, configs):
#         super(ResBlock, self).__init__()
#
#         self.temporal = nn.Sequential(
#             nn.Linear(configs.seq_len, configs.d_model),
#             nn.ReLU(),
#             nn.Linear(configs.d_model, configs.seq_len),
#             nn.Dropout(configs.dropout)
#         )
#
#         self.channel = nn.Sequential(
#             nn.Linear(configs.enc_in, configs.d_model),
#             nn.ReLU(),
#             nn.Linear(configs.d_model, configs.enc_in),
#             nn.Dropout(configs.dropout)
#         )
#
#     def forward(self, x):
#         # x: [B, L, D]
#         x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
#         x = x + self.channel(x)
#         return x
#
#
# class MultiScaleConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(MultiScaleConv1D, self).__init__()
#         self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv1d(in_channels, out_channels, kernel_size=4, padding=2)
#         self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=3)
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.activation = nn.ReLU()
#
#     def forward(self, x):
#         # Apply convolution with different kernel sizes
#         c3 = self.bn(self.activation(self.conv3(x)))
#         c5 = self.bn(self.activation(self.conv4(x)))
#         c7 = self.bn(self.activation(self.conv5(x)))
#         # Concatenate along feature dimension
#         out = c3 + c5 + c7  # element-wise sum to combine features
#         return out
#
#
# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.layer = configs.e_layers
#
#         # Initialize multi-scale convolutional layers
#         self.multi_scale_conv = MultiScaleConv1D(configs.enc_in, configs.enc_in)
#
#         self.model = nn.ModuleList([ResBlock(configs) for _ in range(self.layer)])
#         self.pred_len = configs.pred_len
#         self.projection = nn.Linear(configs.seq_len, configs.pred_len)
#
#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         # Apply multi-scale convolution and activation
#         x_enc = x_enc.transpose(1, 2)  # Reorder dimensions to [B, D, L] for convolution
#         x_enc = self.multi_scale_conv(x_enc)
#         x_enc = x_enc.transpose(1, 2)  # Reorder back to [B, L, D] for further processing
#
#         # Process with ResBlocks
#         for i in range(self.layer):
#             x_enc = self.model[i](x_enc)
#         enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
#
#         return enc_out
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#         return dec_out[:, -self.pred_len:, :]  # Select the output for the prediction length
#
#
# class Config:
#     def __init__(self):
#         self.seq_len = 12
#         self.pred_len = 1
#         self.e_layers = 2
#         self.enc_in = 10
#         self.d_model = 32
#         self.dropout = 0.1
#         self.conv_channels = 16
#
#
# configs = Config()
# model = Model(configs)
#
# # Example tensor for demonstration
# example_input = torch.randn(10, configs.seq_len, configs.enc_in)
# example_output = model(example_input, None, None, None)
# print(example_output.shape)  # Output shape


#
#
#因果卷积
#
# import torch
# import torch.nn as nn
#
#
# class ResBlock(nn.Module):
#     def __init__(self, configs):
#         super(ResBlock, self).__init__()
#
#         self.temporal = nn.Sequential(
#             nn.Linear(configs.seq_len, configs.d_model),
#             nn.ReLU(),
#             nn.Linear(configs.d_model, configs.seq_len),
#             nn.Dropout(configs.dropout)
#         )
#
#         self.channel = nn.Sequential(
#             nn.Linear(configs.enc_in, configs.d_model),
#             nn.ReLU(),
#             nn.Linear(configs.d_model, configs.enc_in),
#             nn.Dropout(configs.dropout)
#         )
#
#     def forward(self, x):
#         # x: [B, L, D]
#         x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
#         x = x + self.channel(x)
#         return x
#
#
# class DilatedCausalConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DilatedCausalConv1D, self).__init__()
#         # Setting up dilated causal convolutions
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
#         self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.activation = nn.ReLU()
#
#     def forward(self, x):
#         # Apply dilated causal convolutions
#         x = self.bn(self.activation(self.conv1(x)))
#         x = self.bn(self.activation(self.conv2(x)))
#         return x
#
#
# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.layer = configs.e_layers
#
#         # Initialize dilated causal convolutional layers
#         self.dilated_causal_conv = DilatedCausalConv1D(configs.enc_in, configs.enc_in)
#
#         self.model = nn.ModuleList([ResBlock(configs) for _ in range(self.layer)])
#         self.pred_len = configs.pred_len
#         self.projection = nn.Linear(configs.seq_len, configs.pred_len)
#
#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         # Apply dilated causal convolution and activation
#         x_enc = x_enc.transpose(1, 2)  # Reorder dimensions to [B, D, L] for convolution
#         x_enc = self.dilated_causal_conv(x_enc)
#         x_enc = x_enc.transpose(1, 2)  # Reorder back to [B, L, D] for further processing
#
#         # Process with ResBlocks
#         for i in range(self.layer):
#             x_enc = self.model[i](x_enc)
#         enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
#
#         return enc_out
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#         return dec_out[:, -self.pred_len:, :]  # Select the output for the prediction length
#
#
# class Config:
#     def __init__(self):
#         self.seq_len = 12
#         self.pred_len = 1
#         self.e_layers = 2
#         self.enc_in = 10
#         self.d_model = 32
#         self.dropout = 0.1
#         self.conv_channels = 16
#
#
# configs = Config()
# model = Model(configs)
#
# # Example tensor for demonstration
# example_input = torch.randn(10, configs.seq_len, configs.enc_in)
# example_output = model(example_input, None, None, None)
# print(example_output.shape)  # Output shape
#
#
#

#深度可分离
# #
#
# import torch
# import torch.nn as nn
#
#
# class ResBlock(nn.Module):
#     def __init__(self, configs):
#         super(ResBlock, self).__init__()
#
#         self.temporal = nn.Sequential(
#             nn.Linear(configs.seq_len, configs.d_model),
#             nn.ReLU(),
#             nn.Linear(configs.d_model, configs.seq_len),
#             nn.Dropout(configs.dropout)
#         )
#
#         self.channel = nn.Sequential(
#             nn.Linear(configs.enc_in, configs.d_model),
#             nn.ReLU(),
#             nn.Linear(configs.d_model, configs.enc_in),
#             nn.Dropout(configs.dropout)
#         )
#
#     def forward(self, x):
#         # x: [B, L, D]
#         x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
#         x = x + self.channel(x)
#         return x
#
#
# class DepthwiseSeparableConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
#         super(DepthwiseSeparableConv1D, self).__init__()
#         # Depthwise convolution
#         self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
#                                         padding=padding, groups=in_channels)
#         # Pointwise convolution
#         self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.activation = nn.ReLU()
#
#     def forward(self, x):
#         x = self.depthwise_conv(x)
#         x = self.pointwise_conv(x)
#         x = self.bn(x)
#         x = self.activation(x)
#         return x
#
#
# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.layer = configs.e_layers
#
#         # Initialize depthwise separable convolutional layers
#         self.depthwise_separable_conv = DepthwiseSeparableConv1D(configs.enc_in, configs.enc_in)
#
#         self.model = nn.ModuleList([ResBlock(configs) for _ in range(self.layer)])
#         self.pred_len = configs.pred_len
#         self.projection = nn.Linear(configs.seq_len, configs.pred_len)
#
#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         # Apply depthwise separable convolution and activation
#         x_enc = x_enc.transpose(1, 2)  # Reorder dimensions to [B, D, L] for convolution
#         x_enc = self.depthwise_separable_conv(x_enc)
#         x_enc = x_enc.transpose(1, 2)  # Reorder back to [B, L, D] for further processing
#
#         # Process with ResBlocks
#         for i in range(self.layer):
#             x_enc = self.model[i](x_enc)
#         enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
#
#         return enc_out
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#         return dec_out[:, -self.pred_len:, :]  # Select the output for the prediction length
#
#
# class Config:
#     def __init__(self):
#         self.seq_len = 12
#         self.pred_len = 1
#         self.e_layers = 2
#         self.enc_in = 10
#         self.d_model = 32
#         self.dropout = 0.1
#         self.conv_channels = 16
#
#
# configs = Config()
# model = Model(configs)
#
# # Example tensor for demonstration
# example_input = torch.randn(10, configs.seq_len, configs.enc_in)
# example_output = model(example_input, None, None, None)
# print(example_output.shape)  # Output shape

#
#
#
#空洞卷积
#
# #
# import torch
# import torch.nn as nn
#
# class ResBlock(nn.Module):
#     def __init__(self, configs):
#         super(ResBlock, self).__init__()
#
#         self.temporal = nn.Sequential(
#             nn.Linear(configs.seq_len, configs.d_model),
#             nn.ReLU(),
#             nn.Linear(configs.d_model, configs.seq_len),
#             nn.Dropout(configs.dropout)
#         )
#
#         self.channel = nn.Sequential(
#             nn.Linear(configs.enc_in, configs.d_model),
#             nn.ReLU(),
#             nn.Linear(configs.d_model, configs.enc_in),
#             nn.Dropout(configs.dropout)
#         )
#
#     def forward(self, x):
#         # x: [B, L, D]
#         x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
#         x = x + self.channel(x)
#         return x
#
# class DilatedCausalConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels, configs):
#         super(DilatedCausalConv1D, self).__init__()
#         # Dilated convolutions with increasing dilation factors
#         self.convolutions = nn.ModuleList([
#             nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
#             for dilation in [1, 2, 4]  # Example dilation factors: 1, 2, 4
#         ])
#         self.activation = nn.ReLU()
#         self.bn = nn.BatchNorm1d(out_channels)  # Batch normalization
#
#     def forward(self, x):
#         # Applying dilated convolutions and summing the outputs
#         output = 0
#         for conv in self.convolutions:
#             output += self.bn(self.activation(conv(x)))
#         return output
#
# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.layer = configs.e_layers
#
#         # Initialize dilated causal convolutional layers
#         self.dilated_causal_conv = DilatedCausalConv1D(configs.enc_in, configs.enc_in, configs)
#
#         self.model = nn.ModuleList([ResBlock(configs) for _ in range(self.layer)])
#         self.pred_len = configs.pred_len
#         self.projection = nn.Linear(configs.seq_len, configs.pred_len)
#
#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         # Apply dilated causal convolution and activation
#         x_enc = x_enc.transpose(1, 2)  # Reorder dimensions to [B, D, L] for convolution
#         x_enc = self.dilated_causal_conv(x_enc)
#         x_enc = x_enc.transpose(1, 2)  # Reorder back to [B, L, D] for further processing
#
#         # Process with ResBlocks
#         for i in range(self.layer):
#             x_enc = self.model[i](x_enc)
#         enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
#
#         return enc_out
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#         return dec_out[:, -self.pred_len:, :]  # Select the output for the prediction length
#
# class Config:
#     def __init__(self):
#         self.seq_len = 12
#         self.pred_len = 1
#         self.e_layers = 5
#         self.enc_in = 10
#         self.d_model = 32
#         self.dropout = 0.1
#
# configs = Config()
# model = Model(configs)
#
# # Example tensor for demonstration
# example_input = torch.randn(10, configs.seq_len, configs.enc_in)
# example_output = model(example_input, None, None, None)
# print(example_output.shape)  # Output shape



#通道混合卷积
#

import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)
        return x

class ChannelMixingConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelMixingConv1D, self).__init__()
        # Convolution layers for each channel individually
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=3, padding=1) for _ in range(in_channels)
        ])
        # Additional convolution to mix the channels
        self.mix_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        outputs = []
        # Apply convolutions separately to each channel
        for i, conv in enumerate(self.conv_layers):
            output = conv(x[:, i:i+1, :])  # Apply conv to each channel
            outputs.append(output)
        # Stack all channel outputs
        outputs = torch.cat(outputs, dim=1)
        # Mix channels
        x = self.mix_conv(outputs)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.layer = configs.e_layers

        # Initialize channel mixing convolutional layers
        self.channel_mixing_conv = ChannelMixingConv1D(configs.enc_in, configs.enc_in)

        self.model = nn.ModuleList([ResBlock(configs) for _ in range(self.layer)])
        self.pred_len = configs.pred_len
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Apply channel mixing convolution and activation
        x_enc = x_enc.transpose(1, 2)  # Reorder dimensions to [B, D, L] for convolution
        x_enc = self.channel_mixing_conv(x_enc)
        x_enc = x_enc.transpose(1, 2)  # Reorder back to [B, L, D] for further processing

        # Process with ResBlocks
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)

        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # Select the output for the prediction length

class Config:
    def __init__(self):
        self.seq_len = 12
        self.pred_len = 1
        self.e_layers = 2
        self.enc_in = 10
        self.d_model = 32
        self.dropout = 0.1

configs = Config()
model = Model(configs)

# Example tensor for demonstration
example_input = torch.randn(10, configs.seq_len, configs.enc_in)
example_output = model(example_input, None, None, None)
print(example_output.shape)  # Output shape




