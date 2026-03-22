import toml
import json
import sys
import pydantic
from loguru import logger
from pprint import pformat
from openai import OpenAI


class GoalAndRule(pydantic.BaseModel):
    goal: str
    rule: list[str]

    def parse(self)->str:
        rslt = f"目標:{self.goal}\n"
        for i, rule in enumerate(self.rule):
            rslt += f"制約{i+1}:{rule}\n"    
        return rslt
    
    def __str__(self):
        return self.parse()

def test_parse():
    goals_and_rules = GoalAndRule(
        goal="空を自由に飛びたいです",
        rule=["世界で人が載れるような小型ドローンの開発が進んでいます", "物理的にある空間に入れられる物の容積は限られています"],
    )
    # print(goals_and_rules.parse())
    assert goals_and_rules.parse() == "目標:空を自由に飛びたいです\n制約1:世界で人が載れるような小型ドローンの開発が進んでいます\n制約2:物理的にある空間に入れられる物の容積は限られています\n"


def 設定():
    logger.info("APIキーを取得しました。")
    with open(r"C:\Users\音声入力用アカウント\Documents\機密情報.toml", "rb") as file:
     data = toml.load(file)
    api_key = data["DEEPSEEK_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return client

client = 設定()

def get_json_answer(messages):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,  # type: ignore
        # https://api-docs.deepseek.com/quick_start/parameter_settings
        # を参考にコーディングさせたいので0.0に設定
        temperature=0.0,
        stream=False,
        response_format={"type": "json_object"},
    )
    return response

def make_system_message(message: str):
    return [{"role": "system", "content": message}]


def make_user_message(message: str):
    return [{"role": "user", "content": message}]

def モデルへ質問(system_message:str, 作業名:str, goals_and_rules: GoalAndRule):
    logger.info(f"モデルへ質問します: {作業名}")
    logger.info(f"{system_message=}")
    formatted_system_message = make_system_message(system_message)
    user_message = goals_and_rules.parse()
    logger.info(f"{user_message=}")
    user_message = make_user_message(user_message)

    response = get_json_answer(formatted_system_message + user_message)
    logger.info(f"モデルの回答は下記でした:\n{response}")
    
    try:
        response = json.loads(response.choices[0].message.content) # type: ignore
        logger.info(f"{作業名}に成功しました")
        return response
    except (json.JSONDecodeError, KeyError, TypeError) as エラー:
        logger.error(f"タスクの分割に失敗しました: {エラー}")
        raise ValueError("タスクの分割に失敗しました。応答の形式が正しくありません。")
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        raise ValueError("タスクの分割に失敗しました。予期しないエラーが発生しました。")

prompt案 = """
あなたはソフトウェアシステムを作るAIエージェントの一部です.
ソフトウェアシステムは多分に複雑になることが予想されるため,必要であればユーザーへ質問,提案や拒否,承諾,ができます.
"""

def 無理か判定(goals_and_rules: GoalAndRule):
    system_message = """
    ユーザーからシステム作成について目標と制約が与えられます.
    あなたはそれらを見て,無理かどうかとその理由を次のJSON形式で出力してください.
    { "判定値": int, "理由": string }
    "判定値"は0なら無理, 1なら可能, 不明瞭な点がある場合は2としてください.
    "理由"はその判定値になった理由を示してください.もし不明瞭な点がある場合はその不明瞭な点を
    解決するような質問も含めてください.

    入力例1: 
        目標:pythonを使用して簡単なapiサーバーを作りたいです.
        制約1:ただしapiサーバーは安定した動作をさせたいです
        制約2:フレームワークはfastapiを使用したいです.
        制約3:データベースはsqliteを使用したいです.

    出力例1:
        { "判定値": 1, "理由": "目標と制約に矛盾はなく,pythonを使用して簡単なapiサーバーを作ることは可能です." }
        
    入力例2: 
        目標:pythonを使用して簡単なapiサーバーを作りたいです.
        制約1:ただしapiサーバーは安定した動作をさせたいです
        制約2:フレームワークはfastapiを使用したいです.
        制約3:データベースはsqliteを使用したいです.
        制約4:言語はGoを使用したいです.

    出力例2: 
        { "判定値": 0, "理由": "目標1と制約4が矛盾しています." }
    """
    return モデルへ質問(system_message, "無理か判定", goals_and_rules)

def 分割が必要か判定(goals_and_rules: GoalAndRule):
    system_message = """
    ユーザーからシステム作成について目標と制約が与えられます.
    あなたはその目標と制約を達成するテストと実行コードできるか,それとも機能の分割が
    必要かを判定してください.
    判定の結果は次のJSON形式で出力してください
    { "判定値": int, "理由": string }
    "判定値"は0なら分割が必要, 1なら分割は不要, 不明瞭な点がある場合は2としてください.
    "理由"はその判定値になった理由を示してください.もし不明瞭な点がある場合はその不明瞭な点を解決するような質問も含めてください.

    入力例1: 
        目標1:pythonを使用して簡単なapiサーバーを作りたいです.
        目標2:ただしapiサーバーは安定した動作をさせたいです
        制約2:フレームワークはfastapiを使用したいです.
        制約3:データベースはsqliteを使用したいです.

    出力例1:
        { "判定値": 0, "理由": "少なくともSQLアクセスのコードとapi処理のコードを分割する必要があります." }
        
    入力例2: 
        目標1:pythonを使用してフィボナッチ数列を作成するコードを作成したいです.
        制約1:中間状態をloggerでlogを残す必要があります.

    出力例2: 
        { "判定値": 1, "理由": "フィボナッチ数列を作成する程度の関数であれば簡易に作成することができます.logを残すことも難しくはないでしょう." }
    """
    return モデルへ質問(system_message, "分割が必要か判定", goals_and_rules)

def タスクを分割(goals_and_rules: GoalAndRule):
    
    system_message内容 = """
    ユーザーからシステム作成についての目標と制約が与えられます.あなたはこれらを複数の小目標に分割してください. 
    分割をする理由はでよりタスクに取り組みやすくするためです.
    分割は2~10個の分割でよいので,もとの目標と制約に対してMECEになるようにしてください.

    各分割された目標と制約は次のJSON形式で出力してください:
    [目標1, 目標2, ...]
    ここで目標の数

    入力例1:
        目標:CRUDができるアプリを実装したい

    出力例1:
        {
            'goals': [
                "データベース層を作成する",
                "データベース操作層を作成する",
                "モデル層を作成する",
                "ユーザーインターフェース層を作成する",
            ]
        }
    """

    return モデルへ質問(system_message内容, "タスクの分割", goals_and_rules)
    
def 分割作業(goal_and_rule: GoalAndRule):
    出力 = タスクを分割(goal_and_rule)
    for タスク in 出力["goals"]:
        logger.debug(f"タスクを分割します: {タスク}")
        返答, 内容 = agent(タスク)
        if 返答 == "達成不可":
            return "達成不可", 内容
        if 返答 == "質問":
            return "質問", 内容
        if 返答 == "分割は不要":
            return "分割は不要", 内容

def 分割判定フロー(goal_and_rule: GoalAndRule):
    判定結果 = 分割が必要か判定(goal_and_rule)
    分割が必要かどうか = 判定結果["判定値"]
    理由 = 判定結果["理由"]
    if 分割が必要かどうか == 0:
        return 理由
    if 分割が必要かどうか == 1: 
        return 分割作業(goal_and_rule)
    if 分割が必要かどうか == 2:
        return "不明瞭な点があります"
    return 理由 if 分割が必要かどうか == 0 else 分割作業(goal_and_rule)

def 実現性判定フロー(goal_and_rule: GoalAndRule):
    判定結果 = 無理か判定(goal_and_rule)
    無理かどうか = 判定結果["判定値"]
    理由 = 判定結果["理由"]
    if 無理かどうか == 0:
        return "達成不可", 理由
    if 無理かどうか == 1:
        return 分割判定フロー(goal_and_rule)
    if 無理かどうか == 2:
        return "質問", 理由


def agent(goal_and_rule: GoalAndRule):
    return 実現性判定フロー(goal_and_rule)

if __name__ == "__main__":
    goals_and_rules = GoalsAndRules(
        goals=["空を自由に飛びたいです"],
        rules=["世界で人が載れるような小型ドローンの開発が進んでいます", "物理的にある空間に入れられる物の容積は限られています"],
    )
    print(無理か判定(goals_and_rules))

    goals_and_rules = GoalsAndRules(
        goals=["x^3+1を最小化する"],
        rules=["x>=0", "x<=10"],
    )
    print(無理か判定(goals_and_rules)
    goals_and_rules = GoalsAndRules(
        goals=["簡単なapiサーバーを作りたいです", "ただしapiサーバーは安定した動作をさせたいです"],
        rules=["言語はpythonを使用したいです", "フレームワークはfastapiを使用したいです", "データベースはsqliteを使用したいです"],
    )
    print(無理か判定(goals_and_rules))


    goals_and_rules = GoalAndRule(
        goal="linuxのような軽量OSを作りたいです",
        rule=["言語はzigを使用したいです", "最新のハードウェアアーキテクチャに対応したいです"],
    )
    print(分割が必要か判定(goals_and_rules))
    print(タスクを分割(goals_and_rules))

