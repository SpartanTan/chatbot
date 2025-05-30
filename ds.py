import os
import argparse
import re

from openai import OpenAI
from prompt_toolkit import PromptSession
from datetime import datetime

from difflib import SequenceMatcher

from colorama import init, Fore, Style
init(autoreset=True)

class ChatSession:
    def __init__(self, api_key=None, base_url="https://api.deepseek.com", model="deepseek-chat", system_message="You are a helpful assistant.", cost=False):
        """
        参数：
        - api_key (str): 平台的 API Key，默认从环境变量 `DEEPSEEK_API_KEY` 读取
        - base_url (str): API 请求地址，默认为 DeepSeek 官方平台
        - model (str): 模型名称（如 'deepseek-chat' 或 'deepseek-reasoner'），默认为 'deepseek-chat'
        - system_message (str): 系统消息，用于设定对话背景，默认为 'You are a helpful assistant.'
        """
        # 处理 API Key 优先级：显式传入 > 环境变量
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("API Key 未提供，请通过参数传入或设置环境变量 DEEPSEEK_API_KEY")
        self.base_url = base_url

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self.model = model
        self.messages = [{'role': 'system', 'content': system_message}]
        self.cost = cost

    def append_message(self, role, content):
        """
        添加一条对话消息

        参数:
        - role (str): 消息角色，为 'user' 或 'assistant'
        - content (str): 消息内容
        """
        self.messages.append({'role': role, 'content': content})

    def get_response(self, user_input, stream=False):
        """
        添加用户消息，调用 API 获取回复，并返回推理过程和回复内容

        参数：
        - user_input (str): 用户输入的消息
        - stream (bool): 是否启用流式输出，默认为 False

        返回：
        if stream=False:
            tuple: (reasoning_content, content)
            - reasoning_content (str|None): 推理过程，仅推理模型返回，聊天模型为 None
            - content (str): 模型的回复内容

        if stream=True:
            generator: 生成一系列 (reasoning_content, content) 元组
            - 对于推理过程: (reasoning_content, None)
            - 对于回复内容: (None, content)
            其中必定有一个值为 None，另一个包含当前数据块的实际内容
        """
        # 记录用户输入
        self.append_message('user', user_input)

        # 调用 API
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=stream
        )

        if not stream:
            # 非流式输出
            content = completion.choices[0].message.content
            reasoning_content = getattr(
                completion.choices[0].message, 'reasoning_content', None)

            # 记录模型回复
            self.append_message('assistant', content)

            return reasoning_content, content
        else:
            # 流式输出，返回生成器
            return self._process_stream(completion)

    def _process_stream(self, completion):
        """
        处理流式输出的数据块

        参数：
        - completion: API 返回的流式输出对象

        返回：
        generator: 生成器对象，每次返回 (reasoning_content, content) 元组
        - 当收到推理过程时: yield (reasoning_content, None)
        - 当收到回复内容时: yield (None, content)
        """
        content = ""  # 用于存储完整回复
        reasoning_printed = False  # 标记是否已经打印过推理过程

        for chunk in completion:
            delta = chunk.choices[0].delta
            # 处理推理过程（仅推理模型有）
            if getattr(delta, 'reasoning_content', None):
                if not reasoning_printed:
                    yield "==Reasoning==\n", None  # 只在推理过程开始时打印一次
                    reasoning_printed = True
                yield delta.reasoning_content, None
            # 处理回复内容
            elif delta.content:
                content += delta.content  # 需要记录 content 维护对话历史
                yield None, delta.content

            # 如果是最后一个数据块（finish_reason 不为 None）
            if chunk.choices[0].finish_reason is not None:
                # 记录完整的模型回复 content
                if self.cost:
                    print_chat_usage(chunk)
                self.append_message('assistant', content)
                break


from wcwidth import wcswidth

def print_chat_usage(completion):
    stats = completion.usage
    hit = stats.prompt_cache_hit_tokens
    miss = stats.prompt_cache_miss_tokens

    input_tokens = stats.prompt_tokens
    output_tokens = stats.completion_tokens
    total_tokens = stats.total_tokens

    input_cost = (hit * 0.5 + miss * 2) / 1_000_000
    output_cost = output_tokens * 8 / 1_000_000
    total_cost = input_cost + output_cost

    print("\n\n")

    entries = [
        ("TOKEN 消耗与成本统计", ""),
        ("输入 Token 数",       f"{input_tokens}（缓存命中: {hit}, 未命中: {miss}）"),
        ("输出 Token 数",       f"{output_tokens}"),
        ("总 Token 数",         f"{total_tokens}"),
        ("", ""),
        ("输入成本",           f"￥{input_cost:.4f} 元"),
        ("输出成本",           f"￥{output_cost:.4f} 元"),
        ("预估总成本",         f"￥{total_cost:.4f} 元")
    ]

    # 构造输出行，计算实际显示宽度
    content_lines = []
    max_display_width = 0
    for left, right in entries:
        if left == "" and right == "":
            line = ""
        elif right:
            line = f"{left} : {right}"
        else:
            line = left
        content_lines.append(line)
        max_display_width = max(max_display_width, wcswidth(line))

    # 打印顶部线（按显示宽度 + 左边两个空格缩进 + 右边两个空格）
    line_width = max_display_width + 2 * 2
    print(f"╭{'─' * line_width}╮")

    for line in content_lines:
        if line.strip() == "":
            print()
        else:
            print(f"  {line}")

    print(f"╰{'─' * line_width}╯")


def get_multiline_input(prompt="💬 (Shift+Enter 换行，Enter 发送)：\n"):
    """
    支持 Shift+Enter 插入换行，Enter 发送
    """
    session = PromptSession()
    return session.prompt(prompt, multiline=True)

def create_session_log_file(directory=None):
    if directory is None:
        # 固定为 ds.py 所在目录下的 "history"
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")

    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(directory, f"{timestamp}.session")

def append_to_log(file_path, role, content):
    now = datetime.now().strftime("%H:%M:%S")
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"[{role} @ {now}]\n{content.strip()}\n\n")

def is_fuzzy_match(line, keyword, threshold=0.6):
    line_lower = line.lower()
    keyword_lower = keyword.lower()

    if keyword_lower in line_lower:
        return True

    # 分词匹配（更丰富的符号）
    tokens = re.split(r'[\s_\-./\\:()\'"`\[\]{}<>]+', line_lower)
    if any(keyword_lower in token for token in tokens):
        print("HIT")
        return True

    # fallback: fuzzy
    return SequenceMatcher(None, line_lower, keyword_lower).ratio() >= threshold

def search_history(keyword, max_results=10):
    from difflib import SequenceMatcher
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")
    print(f"🔍 正在模糊搜索关键词: \"{keyword}\" ...\n")

    results = []
    current_block = ""
    current_header = ""

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".session"):
            continue
        path = os.path.join(directory, filename)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("[User") or line.startswith("[Assistant"):
                # 新段落起始，先处理上一段
                if current_block.strip():
                    if is_fuzzy_match(current_block, keyword):
                        results.append((filename, current_header.strip(), current_block.strip()))
                current_header = line
                current_block = ""
            else:
                current_block += line

        # 文件结束前最后一段也要处理
        if current_block.strip() and is_fuzzy_match(current_block, keyword):
            results.append((filename, current_header.strip(), current_block.strip()))

    if not results:
        print("❌ 没有找到相关记录。")
    else:
        print(f"✅ 找到 {len(results)} 条匹配记录，显示最近 {min(len(results), max_results)} 条：\n")
        for file, header, block in results[-max_results:]:
            print(f"📄 {file} | {header}")
            # print(block)
            print(highlight_keyword(block, keyword))
            print("─" * 50)

def highlight_keyword(text, keyword):
    keyword_lower = keyword.lower()
    result = ""
    i = 0
    while i < len(text):
        if text[i:i+len(keyword)].lower() == keyword_lower:
            result += Fore.RED + Style.BRIGHT + text[i:i+len(keyword)] + Style.RESET_ALL
            i += len(keyword)
        else:
            result += text[i]
            i += 1
    return result

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="DeepSeek Chatbot")
        parser.add_argument('-c', '--cost', action='store_true',
                            help="打印 token 消耗明细和成本")
        parser.add_argument('--search', type=str, help="搜索历史记录中的关键词")
        args = parser.parse_args()

        config = {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "system_message": "You are a helpful assistant.",
            "cost": args.cost
        }

        if args.search:
            search_history(args.search)
            exit(0)

        session = ChatSession(**config)
        log_file = create_session_log_file()

        consecutive_interrupts = 0  # 记录连续中断次数

        while True:
            try:
                while True:
                    try:
                        user_input = get_multiline_input("💬: ")
                        consecutive_interrupts = 0  # 一旦输入成功，重置
                    except KeyboardInterrupt:
                        consecutive_interrupts += 1
                        if consecutive_interrupts >= 2:
                            raise  # 向上层抛出异常，触发退出
                        print("\n🚪 已中断输入（再次 Ctrl+C 退出程序）")
                        continue

                    append_to_log(log_file, "User", user_input)

                    stream = True
                    file_refs = re.findall(r'@file\((.*?)\)', user_input)
                    file_not_found = False

                    for file_name in file_refs:
                        try:
                            with open(file_name, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                                user_input = user_input.replace(
                                    f'@file({file_name})',
                                    f"\n===== 文件 {file_name} 内容如下 =====\n{file_content}\n===== 结束 =====\n"
                                )
                                print(f"📂 Reading file {file_name}...")
                        except FileNotFoundError:
                            print(f"❌ 文件未找到: {file_name}")
                            file_not_found = True
                            break

                    if not file_not_found:
                        break

                print("🤖: ", end='', flush=True)
                reply_accum = ""
                for reasoning, reply in session.get_response(user_input, stream=stream):
                    if reasoning:
                        print(reasoning, end='', flush=True)
                    else:
                        print(reply, end='', flush=True)
                        reply_accum += reply
                append_to_log(log_file, "Assistant", reply_accum)
                print()
            except KeyboardInterrupt:
                consecutive_interrupts += 1
                if consecutive_interrupts >= 2:
                    raise
                print("\n🚪 中断当前回复流程（再次 Ctrl+C 退出程序）")

    except KeyboardInterrupt:
        print("\n👋 已退出程序，再见！")

