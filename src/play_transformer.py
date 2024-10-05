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
        
        # RoPEの適用
        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat([x1 * angle_rads[..., ::2] - x2 * angle_rads[..., 1::2],
                       x1 * angle_rads[..., 1::2] + x2 * angle_rads[..., ::2]], dim=-1)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, mbed_dim, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, mbed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, mbed_dim, 2).float() * (-math.log(10000.0) / mbed_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

class QKLayerNorm(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.ln_q = nn.LayerNorm(head_dim)
        self.ln_k = nn.LayerNorm(head_dim)

    def forward(self, q, k):
        return self.ln_q(q), self.ln_k(k)

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim  # モデル全体の埋め込み次元
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads  # 各ヘッドの次元
        
        # 線形変換層: 埋め込み次元から各ヘッドの入力を生成
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.qk_layernorm = QKLayerNorm(self.head_dim)

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 入力を各ヘッドに分割
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # qk-layernormの適用
        q, k = self.qk_layernorm(q, k)
        
        # scaled_dot_product_attentionを使用
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        # 形状の変更と出力プロジェクション
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        output = self.out_proj(attn_output)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, mbed_dim, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(mbed_dim, d_ff)
        self.linear2 = nn.Linear(d_ff, mbed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class RMSNorm(nn.Module):
    def __init__(self, mbed_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(mbed_dim))

    def forward(self, x):
        # RMSNormの計算
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
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class CustomTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, mbed_dim, num_heads, num_layers, d_ff, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, mbed_dim)
        self.pos_encoder = RotaryPositionalEncoding(mbed_dim)
        self.layers = nn.ModuleList([EncoderLayer(mbed_dim, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = RMSNorm(mbed_dim)
        self.mbed_dim = mbed_dim
        
    def forward(self, src, mask=None):
        x = self.embedding(src) * math.sqrt(self.mbed_dim)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)

class IndependentAdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-15, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(IndependentAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # 状態の初期化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Adam更新
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # 独立重み減衰の適用
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

def train_model(vocab_size, mbed_dim, num_heads, num_layers, d_ff, dropout, num_epochs):
    # モデルの初期化
    model = CustomTransformerEncoder(vocab_size, mbed_dim, num_heads, num_layers, d_ff, dropout)

    # 独立重み減衰を使用したオプティマイザーの初期化
    optimizer = IndependentAdamW(model.parameters(), lr=0.001, weight_decay=0.01, eps=1e-15)

    # 入力テンソルの作成（バッチサイズ1、シーケンス長5000）
    input_tensor = torch.randint(0, vocab_size, (1, 5000))

    # 損失関数の定義（例：クロスエントロピー）
    criterion = nn.CrossEntropyLoss()

    # トレーニングループ
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(input_tensor)
        
        # ダミーのターゲットを作成（実際のタスクに応じて適切なターゲットを使用してください）
        target = torch.randint(0, vocab_size, (1, 5000))
        
        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # パラメータ数の計算
    total_params = sum(p.numel() for p in model.parameters())
    print(f"モデルの総パラメータ数: {total_params:,}")

    return model, total_params

# モデルパラメータ
vocab_size = 10
mbed_dim = 512
num_heads = 8
num_layers = 6
d_ff = 2048
dropout = 0.1
num_epochs = 10

# モデルのトレーニングと総パラメータ数の取得
# trained_model, total_params = train_model(vocab_size, mbed_dim, num_heads, num_layers, d_ff, dropout, num_epochs)

# print(f"トレーニング完了。モデルの総パラメータ数: {total_params:,}")

def test_independent_adamw():
    # 簡単な1次元の最適化問題を定義
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor([0.0]))
            self.b = nn.Parameter(torch.tensor([0.0]))

        def forward(self, x):
            return self.a * x + self.b

    # 目標関数: y = 2x + 1
    def target_function(x):
        return 2 * x + 1

    # モデルとオプティマイザーの初期化
    model = SimpleModel()
    optimizer = IndependentAdamW(model.parameters(), lr=0.1, weight_decay=0.01, eps=1e-15)

    # トレーニングループ
    n_epochs = 1000
    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # ランダムな入力データを生成
        x = torch.rand(100)
        y_true = target_function(x)
        
        # モデルの予測
        y_pred = model(x)
        
        # 損失計算
        loss = nn.MSELoss()(y_pred, y_true)
        losses.append(loss.item())
        
        # 逆伝播と最適化
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}, a: {model.a.item():.4f}, b: {model.b.item():.4f}")

    # 結果のプロット
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    x_test = torch.linspace(0, 1, 100)
    y_true = target_function(x_test)
    y_pred = model(x_test).detach()
    plt.plot(x_test, y_true, label='True')
    plt.plot(x_test, y_pred, label='Predicted')
    plt.title('True vs Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Final parameters: a = {model.a.item():.4f}, b = {model.b.item():.4f}")
    print(f"Target parameters: a = 2.0000, b = 1.0000")

# テスト関数の実行
if __name__ == "__main__":
    test_independent_adamw()