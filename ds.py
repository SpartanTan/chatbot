import os
import argparse
import re

from openai import OpenAI
from prompt_toolkit import PromptSession
from datetime import datetime

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

def create_session_log_file(directory="history"):
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(directory, f"{timestamp}.session")

def append_to_log(file_path, role, content):
    now = datetime.now().strftime("%H:%M:%S")
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"[{role} @ {now}]\n{content.strip()}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek Chatbot")
    parser.add_argument('-c', '--cost', action='store_true',
                        help="打印 token 消耗明细和成本")
    args = parser.parse_args()

    config = {
        # "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",  # 可以修改为推理模型，比如 "deepseek-reasoner" deepseek-chat
        "system_message": "You are a helpful assistant.",
        "cost": args.cost
    }

    session = ChatSession(**config)
    log_file = create_session_log_file()

    while True:
        while True:  # 内层循环用于处理输入和文件检查
            user_input = get_multiline_input("💬: ")
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
                    break  # 退出 for 循环，等待重新输入

            if not file_not_found:
                break  # 文件都找到了，继续处理对话

        print("🤖: ", end='', flush=True)
        
        reply_accum = ""
        for reasoning, reply in session.get_response(user_input, stream=stream):
            if reasoning:
                # 不记录 reasoning 到日志
                print(reasoning, end='', flush=True)
            else:
                print(reply, end='', flush=True)
                reply_accum += reply
        append_to_log(log_file, "Assistant", reply_accum)
        print()
