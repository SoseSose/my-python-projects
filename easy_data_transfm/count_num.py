import random
import pyperclip


def random_binary_array():
    binary_array = [random.randint(0, 1) for _ in range(78)]#50,100,75,87,81,78,80,79

    
    binary_string = ''.join(map(str, binary_array))
    
    ones_count = sum(binary_array)
    
    print(f"生成された配列内の1の数: {ones_count}")
    print(f"生成された数字列: {binary_string}")
    print(f"生成された数字列の長さ: {len(binary_string)}")
    
    pyperclip.copy(binary_string)
    print("生成された数字列がクリップボードにコピーされました。")
    
    return binary_string, ones_count

random_string, count_ones = random_binary_array()

