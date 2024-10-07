import torch
import torch.nn as nn
import math
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, mbed_dim, max_len=5000):
        super().__init__()
        self.mbed_dim = mbed_dim
        self.max_len = max_len
        self.theta = 10000 ** (2 * (torch.arange(0, mbed_dim, 2).float()) / mbed_dim)

    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        angle_rates = position / self.theta
        angle_rads = torch.cat([torch.sin(angle_rates), torch.cos(angle_rates)], dim=-1)
        angle_rads = angle_rads.unsqueeze(0).to(x.device)

        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat(
            [
                x1 * angle_rads[..., ::2] - x2 * angle_rads[..., 1::2],
                x1 * angle_rads[..., 1::2] + x2 * angle_rads[..., ::2],
            ],
            dim=-1,
        )
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, mbed_dim, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, mbed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, mbed_dim, 2).float() * (-math.log(10000.0) / mbed_dim)
        )
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, : x.size(1), :].to(x.device)


class QKLayerNorm(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.ln_q = nn.LayerNorm(head_dim)
        self.ln_k = nn.LayerNorm(head_dim)

    def forward(self, q, k):
        return self.ln_q(q), self.ln_k(k)


class GeGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.w = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        x, gate = self.w(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.qk_layernorm = QKLayerNorm(self.head_dim)
        self.geglu = GeGLU(self.head_dim, self.head_dim)

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

        q, k = self.qk_layernorm(q, k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)

        attn_weights = self.geglu(attn_scores)
        attn_weights = F.normalize(attn_weights, p=2, dim=-1)

        attn_output = torch.matmul(attn_weights, v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embedding_dim)
        )
        output = self.out_proj(attn_output)

        return output


class FeedForward(nn.Module):
    def __init__(self, mbed_dim, d_ff):
        super().__init__()
        self.geglu = GeGLU(mbed_dim, d_ff)
        self.linear = nn.Linear(d_ff, mbed_dim)

    def forward(self, x):
        return self.linear(self.geglu(x))


class RMSNorm(nn.Module):
    def __init__(self, mbed_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(mbed_dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return x / (norm + self.eps) * self.scale


class EncoderLayer(nn.Module):
    def __init__(self, mbed_dim, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(mbed_dim, num_heads)
        self.feed_forward = FeedForward(mbed_dim, d_ff)
        self.norm1 = RMSNorm(mbed_dim)
        self.norm2 = RMSNorm(mbed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class GPTLanguageModel(nn.Module):
    """
    GPT言語モデル。

    このモデルは、トークン埋め込み、位置埋め込み、ブロック、最終層正規化、および言語モデルヘッドで構成されています。

    属性:
        embd (nn.Embedding): トークン埋め込みテーブルの次元
    """

    def __init__(
        self,
        vocab_size,
        n_embd: int,
        n_head: int,
        n_layer: int,
        d_ff: int,
        dropout: float = 0.2,
        max_idx_size: int = 256,
    ):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            [EncoderLayer(n_embd, n_head, d_ff, dropout) for _ in range(n_layer)]
        )
        self.final_layer_norm = RMSNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.max_idx_size = max_idx_size

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """

        Args:
            idx (torch.Tensor): int型で(B,T)の入力テンソル
            targets (torch.Tensor, optional): int型で(B,T)のターゲットテンソル。デフォルトはNone。

        Returns:
            _type_: _description_
        """
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.embd(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.max_idx_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


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


if __name__ == "__main__":
    test_independent_adamw()
