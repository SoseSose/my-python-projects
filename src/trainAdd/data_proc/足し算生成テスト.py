import pytest

from trainAdd.data_proc.足し算生成 import (
    n桁の重複なしランダム整数生成,
    二項n桁の足し算の文字列生成,
    二項n桁足し算ファイル生成,
    桁数の汎化,
    多項の足し算の文字列生成,
    多項足し算ファイル生成,
)

bs_tok = "<"
es_tok = ">"

def test_二項n桁の足し算の文字列生成_基本機能():
    result = 二項n桁の足し算の文字列生成(1, 1, 1.0, 1.0)
    lines = result.strip().split("\n")

    assert len(lines) > 0
    for line in lines:
        question, answer = line.split("=")
        num1, num2 = map(int, question.split("+"))
        answer_num = answer.replace(bs_tok, "").replace(es_tok, "")

        assert 0 <= num1 <= 9
        assert 0 <= num2 <= 9

        assert num1 + num2 == answer_num

def test_二項n桁の足し算の文字列生成_2桁():
    result = 二項n桁の足し算の文字列生成(2, 2, 0.1, 0.1)
    lines = result.strip().split("\n")

    for line in lines:
        question, answer = line.split("=")
        num1, num2 = map(int, question.split("+"))
        answer_num = int(answer[1:-1])

        # 2桁の数字であることを確認
        assert 10 <= num1 <= 99
        assert 10 <= num2 <= 99
        assert num1 + num2 == answer_num

def test_二項n桁の足し算の文字列生成_交換法則():
    result = 二項n桁の足し算の文字列生成(1, 1, 1.0, 1.0)
    lines = result.strip().split("\n")

    # 隣り合う行で交換法則が適用されていることを確認
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break

        q1, a1 = lines[i].split("=")
        q2, a2 = lines[i + 1].split("=")

        num1_1, num1_2 = map(int, q1.split("+"))
        num2_1, num2_2 = map(int, q2.split("+"))

        # 数字が入れ替わっていることを確認
        assert num1_1 == num2_2
        assert num1_2 == num2_1

        # 答えが同じであることを確認
        assert a1 == a2

def test_二項n桁の足し算の文字列生成_エラー処理():
    with pytest.raises(ValueError):
        二項n桁の足し算の文字列生成(4, 4, 1.0, 1.0)  # サンプル数が多すぎる場合

def test_二項n桁足し算ファイル生成(tmp_path):
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    生成パラメータ = [
        (1, 1, 1.0, 1.0),
        (2, 1, 1.0, 1.0),
        (2, 2, 0.1, 0.1),
    ]

    二項n桁足し算ファイル生成(test_dir, 生成パラメータ)

    for param in 生成パラメータ:
        digit1, digit2, _, _ = param
        file_path = test_dir / f"{digit1}桁{digit2}桁.txt"
        assert file_path.exists()

        with open(file_path, "r") as f:
            content = f.read()
            assert len(content) > 0

            lines = content.strip().split("\n")
            assert len(lines) > 0
            for line in lines:
                question, answer = line.split("=")
                num1, num2 = map(int, question.split("+"))
                answer_num = int(answer.replace(bs_tok, "").replace(es_tok, ""))

                assert num1 + num2 == answer_num

                assert 10 ** (digit1 - 1) <= num1 < 10**digit1
                assert 10 ** (digit2 - 1) <= num2 < 10**digit2

def test_桁数の汎化(tmp_path):
    train_生成パラメータ = [
        (1, 1, 1.0, 1.0),
        (2, 2, 0.3, 0.3),
    ]
    test_生成パラメータ = [
        (3, 1, 0.1, 1.0),
        (3, 2, 0.1, 0.1),
    ]
    桁数の汎化(train_生成パラメータ, test_生成パラメータ, tmp_path)

    train_path = tmp_path / "桁数の内挿" / "train"
    test_path = tmp_path / "桁数の内挿" / "test"

    assert train_path.exists()
    assert test_path.exists()

    for param in train_生成パラメータ:
        digit1, digit2, _, _ = param
        file_path = train_path / f"{digit1}桁{digit2}桁.txt"
        assert file_path.exists()

    for param in test_生成パラメータ:
        digit1, digit2, _, _ = param
        file_path = test_path / f"{digit1}桁{digit2}桁.txt"
        assert file_path.exists()

def test_n桁の重複なしランダム整数生成_基本機能():
    result = n桁の重複なしランダム整数生成(2, 0.1)
    assert len(result) > 0
    assert len(result) <= 10  # 2桁の数字でサンプル割合0.1の場合、最大10個

    for num in result:
        assert 10 <= num <= 99  # 2桁の数字であることを確認

def test_n桁の重複なしランダム整数生成_重複なしを確認():
    result = n桁の重複なしランダム整数生成(3, 0.05)
    assert len(result) > 0
    assert len(result) <= 45

    assert len(result) == len(set(result))

def test_n桁の重複なしランダム整数生成_サンプル数制限():
    with pytest.raises(ValueError):
        n桁の重複なしランダム整数生成(4, 1.0)  # 4桁の数字でサンプル割合1.0の場合、10000個を超える

def test_多項の足し算の文字列生成_基本機能():
    result = 多項の足し算の文字列生成(3, 0.1)
    lines = result.strip().split("\n")
    assert len(lines) > 0
    for line in lines:
        question, answer = line.split("=")
        terms = question.split("+")
        assert len(terms) == 3
        answer_num = int(answer.replace("<", "").replace(">", ""))
        assert sum(map(int, terms)) == answer_num

def test_多項の足し算の文字列生成_最小項数():
    result = 多項の足し算の文字列生成(2, 0.1)
    lines = result.strip().split("\n")
    assert len(lines) > 0
    for line in lines:
        question, answer = line.split("=")
        terms = question.split("+")
        assert len(terms) == 2
        answer_num = int(answer.replace("<", "").replace(">", ""))
        assert sum(map(int, terms)) == answer_num

def test_多項の足し算の文字列生成_最大項数():
    result = 多項の足し算の文字列生成(10, 0.01)
    lines = result.strip().split("\n")
    assert len(lines) > 0
    for line in lines:
        question, answer = line.split("=")
        terms = question.split("+")
        assert len(terms) == 10
        answer_num = int(answer.replace("<", "").replace(">", ""))
        assert sum(map(int, terms)) == answer_num

def test_多項の足し算の文字列生成_エラー処理():
    with pytest.raises(ValueError):
        多項の足し算の文字列生成(1, 0.1)  # 項数が2未満の場合

    with pytest.raises(ValueError):
        多項の足し算の文字列生成(2, 1.1)  # サンプル割合が1を超える場合

def test_多項足し算ファイル生成(tmp_path):
    dir_path = tmp_path / "test_dir"
    生成パラメータ = [
        (3, 0.1),
        (5, 0.05),
    ]
    多項足し算ファイル生成(dir_path, 生成パラメータ)

    for param in 生成パラメータ:
        term_num, _ = param
        file_path = dir_path / f"{term_num}項.txt"
        assert file_path.exists()