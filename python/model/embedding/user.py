import torch
from torch import nn, Tensor
from typing import Dict

class UserEmbedding( nn.Module ):
    def __init__(self, user_num, hidden_dim, output_dim, **kargs) -> None:
        super().__init__(kargs)
        self.user_embed = nn.Embedding(user_num, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
            Let B = batch size

            inputs
                ['user_id']:              LongTensor( [B, 1] )
                ['gender']:               LongTensor( [B, 1] )
                ex: [ 1, 3, 4 ]

                ['occupation_titles']:    LongTensor( [B, max( length of title ) ] )
                ['interests']:            LongTensor( [B, max( length of interest)] )
                ['groups']:               LongTensor( [B, max( length of group)] )
                ['subgroups']:            LongTensor( [B, max( length of subgroup)] )
                ['recreation_names']:     LongTensor( [B, max( length of recreation)] )
                ex: [
                        [2, 3, PAD],
                        [3, 4, 6],
                        [7, PAD, PAD]
                    ]
        """
        feat_user = self.user_embed( inputs['user_id'] )
        output = self.linear(feat_user)
        return output
