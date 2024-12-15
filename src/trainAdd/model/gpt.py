from typing import Optional, Tuple
import math
import torch
import torch.nn.functional as F
from torch import nn
from model.modules import EncoderLayer, RMSNorm
from loguru import logger



class GPT(nn.Module):
    """
    GPT言語モデル。

    このモデルは、トークン埋め込み、位置埋め込み、ブロック、最終層正規化、および言語モデルヘッドで構成されています。

    属性:
        embd (nn.Embedding): トークン埋め込みテーブルの次元
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        d_ff: int,
        seq_len: int,
        dropout: float,
    ):
        super().__init__()


        self.n_layer = n_layer
        self.embd = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.ModuleList([EncoderLayer(n_embd, n_head, d_ff, seq_len, dropout) for _ in range(n_layer)])
        self.final_layer_norm = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.reset_parameters()

    def reset_parameters(self):
        # https://arxiv.org/pdf/2312.16903 spike no more を参照
        for block in self.blocks:
            std = math.sqrt(2 / 5 / block.mbed_dim)
            scaled_std = std / math.sqrt(2 * self.n_layer)

            nn.init.trunc_normal_(block.self_attn.q_proj.weight, std=std)
            nn.init.trunc_normal_(block.self_attn.k_proj.weight, std=std)
            nn.init.trunc_normal_(block.self_attn.v_proj.weight, std=std)
            nn.init.trunc_normal_(block.self_attn.out_proj.weight, std=scaled_std)
            nn.init.trunc_normal_(block.feed_forward.geglu.gate_w.weight, std=std)
            nn.init.trunc_normal_(block.feed_forward.geglu.x_w.weight, std=std)
            nn.init.trunc_normal_(block.feed_forward.linear.weight, std=scaled_std)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        モデルの順伝播を行います。

        引数:
            idx (torch.Tensor): int型で(B,T)の形状を持つ入力テンソル。
                B はバッチサイズ、T はシーケンス長を表します。
            targets (torch.Tensor, optional): int型で(B,T)の形状を持つターゲットテンソル。
                デフォルトはNoneです。

        戻り値:
            tuple: (logits, loss)
                - logits (torch.Tensor): モデルの出力ロジット。形状は(B,T,vocab_size)です。
                - loss (torch.Tensor or None): targetsが与えられた場合の損失値。
                  与えられなかった場合はNoneです。
        """
        B, T = token_ids.shape

        # idx and targets are both (B,T) tensor of integers
        x = self.embd(token_ids)  # (B,T,C)
        for block in self.blocks:
            x = block(x, mask)  # (B,T,C)
        x = self.final_layer_norm(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


if __name__ == "__main__":
    vocab_size = 100
    model = GPT(
        vocab_size=vocab_size,
        n_embd=128,
        n_head=8,
        n_layer=12,
        d_ff=128,
        seq_len=128,
        dropout=0.1,
    )
    model.reset_parameters()
    
    input = torch.randint(0, vocab_size, (1, 10))
    out = model(input)
    print(out[0].shape)

