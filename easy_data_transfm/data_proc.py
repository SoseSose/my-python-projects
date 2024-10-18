from torch.utils.data import Dataset
from pathlib import Path


#1文字が1トークンにencodeされるとして実装している
#もしbyteベースのencodeにする際は要変更

bs_tok = '<'#bos_token
es_tok = '>'#eos_token
pd_tok = '□'#pad_token

def 足し算ドリルを生成(dir_path:str, limit_num:int):

    with open(dir_path + '足し算ドリル.txt', 'w') as f:
        for i in range(limit_num):
            for j in range(limit_num):
                ques_part = f'{i}+{j}='
                ans_part = f'{bs_tok}{i+j}{es_tok}'
                f.write(f'{ques_part}{ans_part}\n')
    

if __name__ == '__main__':
    足し算ドリルを生成('dataset/', 10)

def 問答を作成(dir_path:str):
    data = Path("dataset/足し算ドリル.txt").read_text()
    問答set:list[str] = data.split('\n')
    dataset = []
    for 問答 in 問答set:
        if 問答 == '':
            break
        bs_tok_pos = 問答.index(bs_tok)+1
        es_tok_pos = 問答.index(es_tok)+1
        q_part = 問答[:bs_tok_pos] #question part
        a_part = 問答[bs_tok_pos:es_tok_pos] #answer part
        ans_len = len(a_part)

        for i in range(ans_len):
            ans = a_part[:i+1]
            ques_and_masked_ans = q_part + ans
            dataset.append(ques_and_masked_ans)

    return dataset

if __name__ == '__main__':
    dataset = 問答を作成('dataset/')
    for data in dataset:
        print(data)
    

class 足し算ドリル(Dataset):
    def __init__(self, seq_len:int):
        self.dataset = 問答を作成('dataset/')
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx][:-1].ljust(self.seq_len, pd_tok)
        y = self.dataset[idx].ljust(self.seq_len, pd_tok)
        return x, y

if __name__ == '__main__':
    dataset = 足し算ドリル(15)
    for data in dataset:
        print(data)

