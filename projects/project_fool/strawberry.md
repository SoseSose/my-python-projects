
## はじめに

2024年9月12日、OpenAIは新しいモデル「GPT4-o1」をリリースしました。  
[OpenAI発表サイト](https://openai.com/index/introducing-openai-o1-preview/)  
このモデルはGPT-4oをさらに進化させたもので、応答する前に考える時間を増やすように  
設計されており推論能力が大幅に強化されています。  
特に科学、コーディング、数学など多段階の推論が必要な複雑な問題での性能向上が注目されています。  
  
実際、GPT4-o1は東京大学の理系数学の入試問題で合格最低点を超える実力を持っているとの[記事](https://metaskilling.blog/chatgpt-o1-toudai-math/)もあります。  
  
面白そうなので「strawberry問題」の拡張版をGPT4-O1に解かせてみることにしました。  

## ストロベリー問題とその拡張

strawberry問題とは、strawberryという単語に「r」がいくつあるかを尋ねたとき  
今までの言語モデルが回答をよく間違えていたことに由来します。  
strawberry問題はそのままGPT4-o1の開発コードとしても使われていたようです。  

## 実験方法

と言ってもstrawberry問題をそのまま拡張することは難しいので、次のように実験しました。

1. まず"次の数字列の中に１がいくつあるか教えて"とプロンプトを与える。
2. 返答の後に0と1が並んだ数値列を与える。
3. 正確に計算できているか確かめる。

この作業を繰り返し、どれだけの長さまでなら1の数を数えられるか
スタートを1000桁として二分探索してみました。

## 実験に使用したコード

下記のコードを使用して、01の文字列をクリップボードにコピーし、chatGPTに貼り付けて作業しました。

```python
import random

import pyperclip

random.seed(1)

correct_num = 0
challenge_num = 1024
wrong_num = 1024


while correct_num != wrong_num - 1:
    binary_array = [random.randint(0, 1) for _ in range(challenge_num)]
    binary_string = "".join(map(str, binary_array))

    ones_count = sum(binary_array)

    print(f"生成された数字列: {binary_string}")
    print(f"生成された数字列の長さ: {len(binary_string)}")
    print(f"生成された配列内の1の数: {ones_count}")
    print(f"correct_num:{correct_num}, wrong_num:{wrong_num}")

    pyperclip.copy(binary_string)
    print("生成された数字列がクリップボードにコピーされました。")

    is_correct = input("正解ですか？不正解ですか？(Y/N)")

    if is_correct == "Y":
        correct_num = challenge_num
    elif is_correct == "N":
        wrong_num = challenge_num
    else:
        print("YかNを入力してください")

    challenge_num = int((wrong_num - correct_num) / 2) + correct_num

```

## 結果

### 1024桁

![1024.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/595608/a573fd16-557e-1598-8752-a64cd0cfe400.png)
[chatGPTリンク](https://chatgpt.com/share/670bb2aa-a8ec-8009-9102-5a37bd9406b8)
正しい答え:524個
結果:

### 512桁

![512.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/595608/c5208a54-79cb-ec48-3944-95f5c812f2c4.png)
[chatGPTリンク](https://chatgpt.com/share/670bb2c6-caf4-8009-a939-553407e24f68)
正しい答え:255個
結果：×

### 256桁

![256.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/595608/06bd2c29-b02b-bb37-d11c-1b33cceb552f.png)
[chatGPTリンク](https://chatgpt.com/share/670bb2dc-3b90-8009-b33e-79cb412ff160)
正しい答え:121個
結果：×

### 128桁

![128.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/595608/e005b33b-3a51-6417-a7a0-6abc101b1c05.png)
[chatGPTリンク](https://chatgpt.com/share/670bb30c-0444-8009-88ff-2f9a518dc372)
正しい答え:66個
結果:×

### 64桁

![64.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/595608/e13c1157-c78c-299d-84cb-74eb3a5366c2.png)
[chatGPTリンク](https://chatgpt.com/share/670bb318-bd54-8009-8725-33abd3764259)
正しい答え:35個
結果:〇

### 96桁

![96.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/595608/da3dacbf-3e67-3e99-53a5-4ab93f4d3a3e.png)
[chatGPTリンク](https://chatgpt.com/share/670bb331-4a78-8009-acca-737397251d64)
正しい答え:56個
結果:×

### 80桁

![80.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/595608/7e17c6f1-66cf-6e38-5d54-5baa0e198649.png)
[chatGPTリンク](https://chatgpt.com/share/670bb359-a490-8009-b226-57e9ed141044)
正しい答え:36個
結果:×

### 72桁

![72.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/595608/3e36446e-3da2-c743-2864-a195268ccd12.png)
[chatGPTリンク](https://chatgpt.com/share/670bb388-7620-8009-af59-719f8a7d2846)
正しい答え:33個
結果:×

### 68桁

![68.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/595608/d0bb71b6-2094-b7a1-7918-e23fc4483609.png)
[chatGPTリンク](https://chatgpt.com/share/67092b04-eae8-8009-92c7-9b03755ad452)
正しい答え:34個
結果:〇


### 70桁

![70.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/595608/e16960c0-21db-c133-c5b5-62939a17b089.png)[chatGPTリンク](https://chatgpt.com/share/670bb3d5-6474-8009-a9da-5095606d68f7)
正しい答え:36個
結果:×

### 69桁

![69.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/595608/0c0282f7-b36a-d8ba-abd1-29ab660f5b93.png)
[chatGPTリンク](https://chatgpt.com/share/670bb3e6-70d0-8009-971a-63aec443216b)
正しい答え:36個
結果:×

結果は**68桁**でした。
なお記事の都合で1巡しただけですが、ほかに何度か実験しても60~80桁くらいの間でした。

## 考察

このように二分探索を使えば、chatGPTがstrawberry問題に対応できる長さがすぐに見つかります！！


といいたいわけではなく..
実はこの実験をしたのは次の論文を読んだからです。
本論文ではTransformerが足し算をするタスクが構造的に苦手だと論じられています。
なお、ほかにもインデックス操作も苦手だとも結論づけています。

[参考論文：What Algorithms can Transformers Learn? A Study in Length Generalization](https://arxiv.org/abs/2310.16028)

人であればこの問題は簡単に解けるように感じます。
1000桁数えろと言われれば、その人を睨むかもしれませんが、気を付ければ数え
られる気がします。まして、100桁ならほぼ間違いなく数えられるでしょう。
aiが原理的に人の考え方と異なっているからこの問題が解けないんだと言うつもりはありません。
そのうちaiが人の思考に追いついてくるのは間違いありません。
事実、数学はすでに間違いなく私より上でしょう。
ただ今のaiに使われている技術がこのようなタスクに向いていないのです。

それでも66桁というのは一つのマイルストーンにも思えます。

## まとめ

GPT4-o1と会話するのは確かに楽しいです。
こんな酔狂な使い方をしている人は中々いなそうですね笑
