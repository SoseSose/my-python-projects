from loguru import logger
import sys

def 設定():
    def カスタムフォーマット(record):
        """
        ログレコードを整形するカスタムフォーマット関数。

        Args:
            record (dict): ログレコードの辞書。

        Returns:
            str: 整形されたログメッセージ。
        """

        時間 = record["time"].strftime("%Y-%m-%d %H:%M:%S")
        レベル = record["level"].name
        モジュール名 = record["module"]
        関数名 = record["function"]
        if 関数名 == "<module>":
            関数名 = ""
        行番号 = record["line"]
        メッセージ = record["message"]
        プロセス_id = record["process"].id
        プロセス_name = record["process"].name
        スレッド_id = record["thread"].id
        スレッド_name = record["thread"].name

        return (
            f"<green>{時間}</green>|"
            f"<level>{レベル:^{8}}</level>|"
            f"<cyan>{モジュール名}.{関数名}:{行番号}</cyan>|\n"
            f"<green>process id:{プロセス_id}</green> "
            f"<green>name:{プロセス_name}</green>|\n"
            f"<cyan>thread id:{スレッド_id}</cyan> "
            f"<cyan>name:{スレッド_name}</cyan>|"
            f"\n<level>{メッセージ}</level>\n"
        )
        
    logger.remove()
    logger.add(sys.stderr, level="INFO", format=カスタムフォーマット)


def ログテスト(テスト文字列):
    logger.info(テスト文字列)
    logger.debug(テスト文字列)
    logger.warning(テスト文字列)
    logger.error(テスト文字列)
    logger.critical(テスト文字列)
    logger.exception(テスト文字列)


if __name__ == "__main__":
    logger.remove()
    # 新しいハンドラを追加し、フォーマットを設定
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green>|"
            "<level>{level}</level>|"
            "<cyan>{module}.{function}:{line}</cyan>|\n"
            "<cyan>process name:{process.name}</cyan>|"
            "<cyan>id:{process.id}</cyan>|"
            "<cyan>thread name:{thread.name}</cyan>|"
            "<cyan>id:{thread.id}</cyan>|\n"
            "<level>{message}</level>"
        ),
        level="DEBUG",
    )
    
    ログテスト("ログテスト3")
   