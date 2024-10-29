import os
import chardet
from pathlib import Path

def check_file_encoding(file_path: Path) -> tuple[str, str, float]:
    """
    ファイルのエンコーディングを検出する

    Args:
        file_path: チェックするファイルのパス

    Returns:
        tuple[str, str, float]: (ファイルパス, 検出されたエンコーディング, 信頼度)
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return str(file_path), result['encoding'], result['confidence']

def find_non_utf8_files(directory: str | Path, 
                       exclude_dirs: set[str] = None,
                       exclude_extensions: set[str] = None) -> list[tuple[str, str, float]]:
    """
    指定されたディレクトリ以下のUTF-8以外でエンコードされているファイルを探す

    Args:
        directory: 探索を開始するディレクトリ
        exclude_dirs: 除外するディレクトリ名のセット
        exclude_extensions: 除外するファイル拡張子のセット

    Returns:
        list[tuple[str, str, float]]: 非UTF-8ファイルのリスト (パス, エンコーディング, 信頼度)
    """
    if exclude_dirs is None:
        exclude_dirs = {'.git', '__pycache__', 'node_modules', 'venv','.venv', '.env', 'logs', 'lightning_logs'}
    
    if exclude_extensions is None:
        exclude_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.bin', 
                            '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip'}

    non_utf8_files = []
    directory = Path(directory)

    for root, dirs, files in os.walk(directory):
        # 除外ディレクトリをスキップ
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            file_path = Path(root) / file
            
            # バイナリファイルや除外拡張子をスキップ
            if file_path.suffix.lower() in exclude_extensions:
                continue
            
            try:
                file_path_str, encoding, confidence = check_file_encoding(file_path)
                if encoding and encoding.lower() != 'utf-8':
                    non_utf8_files.append((file_path_str, encoding, confidence))
            except Exception as e:
                print(f"エラー: {file_path} の処理中に問題が発生しました: {e}")

    return non_utf8_files

def main():
    # カレントディレクトリから探索開始
    current_dir = Path.cwd()/"projects"
    print(f"探索開始ディレクトリ: {current_dir}")
    
    non_utf8_files = find_non_utf8_files(current_dir)
    
    if non_utf8_files:
        print("\nUTF-8以外でエンコードされているファイル:")
        for file_path, encoding, confidence in non_utf8_files:
            print(f"ファイル: {file_path}")
            print(f"エンコーディング: {encoding} (信頼度: {confidence:.2f})")
            print("-" * 80)
    else:
        print("\nUTF-8以外でエンコードされているファイルは見つかりませんでした。")

if __name__ == "__main__":
    main() 