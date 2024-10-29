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
