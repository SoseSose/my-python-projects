import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

import matplotlib.pyplot as plt

class RMSNorm(nn.Module):
    def __init__(self, mbed_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(mbed_dim))
    
    def reset_parameters(self):
        nn.init.zeros_(self.scale)

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return x / (norm + self.eps) * self.scale


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, max_seq_len:int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        head_dim = embedding_dim // num_heads
        
        self.head_dim = head_dim

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.q_layernorm = RMSNorm(head_dim)
        self.k_layernorm = RMSNorm(head_dim)

        self.pos_embd = RotaryPositionalEmbeddings(head_dim, max_seq_len)


    def reset_parameters(self):
        nn.init.trunc_normal_(self.q_proj.weight, std=0.02)
        nn.init.trunc_normal_(self.k_proj.weight, std=0.02)

        self.q_layernorm.reset_parameters()
        self.k_layernorm.reset_parameters()

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.size()

        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        q,k = self.pos_embd(q), self.pos_embd(k)
        #https://www.reddit.com/r/LocalLLaMA/comments/1apn1dy/comment/kqg699o/
        #ropeはq,kに対して行う
        q, k = self.q_layernorm(q), self.k_layernorm(k)

        # if attn_mask is not None:
        #     attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  
        #     attn_mask = attn_mask.expand_as(q)#[B, H, T, T]

        output =torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        #flash attentionを使うことも考える。今のpytorch実装ではmaskがあるとflash attentionは使われない。https://zenn.dev/nhandsome/articles/388b2ebb57d5d1参照

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)

        return self.out_proj(output)

class GeGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.w = nn.Linear(input_dim, output_dim * 2)
        self.gate_w = nn.Linear(input_dim, output_dim)
        self.x_w = nn.Linear(input_dim, output_dim)
        #重みなしの実装もある。どちらを使うか？

    def forward(self, x):
        gate = F.silu(self.gate_w(x))
        x = self.x_w(x)
        return x * gate


class FeedForward(nn.Module):
    #https://github.com/google-deepmind/gemma/blob/main/gemma/modules.py
    #参考にしたけど活性化関数の後のlinearがなぜかzeroで初期化されている。
    def __init__(self, mbed_dim, d_ff):
        super().__init__()
        self.geglu = GeGLU(mbed_dim, d_ff)
        self.linear = nn.Linear(d_ff, mbed_dim)

    def forward(self, x):
        return self.linear(self.geglu(x))


class EncoderLayer(nn.Module):
    """
    トランスフォーマーのエンコーダーレイヤー

    このクラスは、自己注意機構とフィードフォワードネットワークを含むエンコーダーレイヤーを実装します。
    Pre-Layer Normalization アーキテクチャを採用しています。

    Args:
        mbed_dim (int): 埋め込みの次元数
        num_heads (int): 注意機構のヘッド数
        d_ff (int): フィードフォワードネットワークの隠れ層の次元数
        seq_len (int): 入力シーケンスの最大長
        dropout (float, optional): ドロップアウト率。デフォルトは0.1

    Attributes:
        mbed_dim (int): 埋め込みの次元数
        self_attn (MultiHeadAttention): マルチヘッド自己注意機構
        feed_forward (FeedForward): フィードフォワードネットワーク
        norm1 (RMSNorm): 第1の正規化層
        norm2 (RMSNorm): 第2の正規化層
        dropout (nn.Dropout): ドロップアウト層
    """

    def __init__(self, mbed_dim:int, num_heads:int, d_ff:int, seq_len:int, dropout=0.1,):
        super().__init__()
        self.mbed_dim = mbed_dim
        self.self_attn: MultiHeadAttention = MultiHeadAttention(mbed_dim, num_heads, seq_len)
        self.feed_forward: FeedForward = FeedForward(mbed_dim, d_ff)
        self.norm1 = RMSNorm(mbed_dim)
        self.norm2 = RMSNorm(mbed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        エンコーダーレイヤーの順伝播

        Args:
            x (torch.Tensor): 入力テンソル、形状は (batch_size, seq_len, mbed_dim)
            mask (torch.Tensor, optional): 注意機構に適用するマスク。デフォルトはNone

        Returns:
            torch.Tensor: 出力テンソル、形状は入力と同じ (batch_size, seq_len, mbed_dim)
        """

        x = self.norm1(x)
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        return x


if __name__ == "__main__":
    x = torch.randn(1, 10, 128)
    model = EncoderLayer(128, 8, 256, 256)
    print(model(x).shape)

class IndependentAdamW(optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-15, weight_decay=1e-2
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(IndependentAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.mul_(1 - group["lr"] * group["weight_decay"])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def test_independent_adamw():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor([0.0]))
            self.b = nn.Parameter(torch.tensor([0.0]))

        def forward(self, x):
            return self.a * x + self.b

    def target_function(x):
        return 2 * x + 1

    model = SimpleModel()
    optimizer = IndependentAdamW(
        model.parameters(), lr=0.1, weight_decay=0.01, eps=1e-15
    )
    n_epochs = 1000
    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        x = torch.rand(100)
        y_true = target_function(x)
        y_pred = model(x)
        loss = nn.MSELoss()(y_pred, y_true)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}, a: {model.a.item():.4f}, b: {model.b.item():.4f}"
            )

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    x_test = torch.linspace(0, 1, 100)
    y_true = target_function(x_test)
    y_pred = model(x_test).detach()
    plt.plot(x_test, y_true, label="True")
    plt.plot(x_test, y_pred, label="Predicted")
    plt.title("True vs Predicted")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Final parameters: a = {model.a.item():.4f}, b = {model.b.item():.4f}")
    print(f"Target parameters: a = 2.0000, b = 1.0000")


# if __name__ == "__main__":
#     test_independent_adamw()

class CosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.00001,
        eta_min: float = 0.00001,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (torch.optim.Optimizer):
                最適化手法インスタンス
            warmup_epochs (int):
                linear warmupを行うepoch数
            max_epochs (int):
                cosine曲線の終了に用いる 学習のepoch数
            warmup_start_lr (float):
                linear warmup 0 epoch目の学習率
            eta_min (float):
                cosine曲線の下限
            last_epoch (int):
                cosine曲線の位相オフセット
        学習率をmax_epochsに至るまでコサイン曲線に沿ってスケジュールする
        epoch 0からwarmup_epochsまでの学習曲線は線形warmupがかかる
        https://pytorch-lightning-bolts.readthedocs.io/en/stable/schedulers/warmup_cosine_annealing.html
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        return None

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

if __name__ == "__main__":
    model = nn.Linear(1,1)
    max_epoch = 128  # 学習終了のepoch数は最初から与える
    iter_step = 4  # (ダミー)ミニバッチで学習する場合のイテレーションステップ数=バッチサイズ/ミニバッチサイズ

    optimizer = torch.optim.AdamW(  
        model.parameters(), 
        lr=0.001,
        weight_decay=0.02)

    lr_scheduler = CosineAnnealingLR(
        optimizer,
        max_epochs=max_epoch,
        warmup_epochs=8,
        warmup_start_lr=0.0001,
        eta_min=0.00001)

    curves = []
    for e in range(max_epoch):
        for s in range(iter_step):
            # 各イテレーションでパラメータを更新
            optimizer.step()
            curves += [optimizer.param_groups[0]["lr"]]
        # 各epoch終了後にスケジューラで最適化学習率を更新
        lr_scheduler.step()

    plt.plot(curves)
    plt.show()