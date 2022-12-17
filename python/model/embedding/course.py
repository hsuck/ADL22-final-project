import torch
from torch import nn, Tensor
from typing import Dict

class CourseEmbedding( nn.Module ):
    def __init__(self, user_num, hidden_dim, output_dim, **kargs) -> None:
        super().__init__(kargs)
        self.course_embed = nn.Embedding(user_num, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
            Let B = batch size

            inputs
                ['course_id']:            LongTensor( [B, 1] )
        """
        feat_course = self.course_embed( inputs['course_id'] )
        output = self.linear(feat_course)
        return output
