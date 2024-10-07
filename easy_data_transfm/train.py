from easy_data_transfm.modules import CustomTransformerEncoder

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


class Tokenizer:
    def __init__(self, text:str):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
    def encode(self, val:str)->list[int]:
        return [self.stoi[c] for c in val]
    
    def decode(self, val:list[int])->str:
        return ''.join([self.itos[i] for i in val])

 
tokenizer = Tokenizer(text)
mbed_dim = 128
num_heads = 8
num_layers = 12
d_ff = 256
model = CustomTransformerEncoder(tokenizer.vocab_size, mbed_dim, num_heads, num_layers, d_ff, dropout=0.1)

