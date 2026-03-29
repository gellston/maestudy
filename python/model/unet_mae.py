
import torch
import torch.nn as nn


import utils.sparse as sparse

from model.convnextv2_mae import ConvNeXtV2_MAE
from model.decoder import Decoder



class Unet_MAE(nn.Module):
    def __init__(
        self,
        encoder:ConvNeXtV2_MAE,
        decoder:Decoder
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.dims = self.encoder.dims

        self.mask_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.dims[i], 1, 1)) for i in range(4)
        ])
            
        
    def forward(self, x):
        features = self.encoder(x)

        dense_features = []
        for i in range(4):
            f = features[i]

            _, _, h, w = f.shape


            m = sparse._get_active_ex_or_ii(h, w, returning_active_ex=True)
            t = self.mask_tokens[i]
            
            # 0인 자리를 해당 스테이지의 마스크 토큰으로 교체
            f_dense = f * m + t * (1 - m)
            dense_features.append(f_dense)

        output = self.decoder(dense_features)

        return output