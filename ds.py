from openai import OpenAI
import readline
import os
import argparse
import re
from pprint import pprint

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys


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


def print_chat_usage(completion):
    stats = completion.usage
    hit = stats.prompt_cache_hit_tokens
    miss = stats.prompt_cache_miss_tokens

    print(f"===== TOKEN 消耗明细 =====")
    print(f"输入: {stats.prompt_tokens} tokens [缓存命中: {hit} | 未命中: {miss}]")
    print(f"输出: {stats.completion_tokens} tokens")
    print(f"总消耗: {stats.total_tokens} tokens")

    input_cost = (hit * 0.5 + miss * 2) / 1_000_000
    output_cost = stats.completion_tokens * 8 / 1_000_000
    total_cost = input_cost + output_cost

    print(f"\n===== 成本明细 =====")
    print(f"输入成本: ￥{input_cost:.4f} 元")
    print(f"输出成本: ￥{output_cost:.4f} 元")
    print(f"预估总成本: ￥{total_cost:.4f} 元")


# def get_multiline_input(prompt="Input:"):
#     """
#     允许用户输入多行，直到按下空行（回车）时结束输入。
#     """
#     lines = []
#     print(prompt)
#     while True:
#         try:
#             line = input()
#             if line.strip() == "":  # 输入空行表示结束输入
#                 break
#             lines.append(line)
#         except EOFError:  # 如果按下 Ctrl+D 结束输入
#             break
#     return "\n".join(lines)


def get_multiline_input(prompt="💬 (Shift+Enter 换行，Enter 发送)：\n"):
    """
    支持 Shift+Enter 插入换行，Enter 发送
    """
    session = PromptSession()
    return session.prompt(prompt, multiline=True)

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

    while True:
        user_input = get_multiline_input("💬: ")
        stream = True  # 非流式输出

        # 检查是否包含 @file(...) 引用
        file_refs = re.findall(r'@file\((.*?)\)', user_input)
        for file_name in file_refs:
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    # 将 @file(...) 替换为实际文件内容
                    user_input = user_input.replace(
                        f'@file({file_name})', f"\n===== 文件 {file_name} 内容如下 =====\n{file_content}\n===== 结束 =====\n")
            except FileNotFoundError:
                print(f"❌ 文件未找到: {file_name}")
                continue

        print("🤖: ", end='', flush=True)
        for reasoning, reply in session.get_response(user_input, stream=stream):
            if reasoning:
                print(reasoning, end='', flush=True)
            else:
                print(reply, end='', flush=True)
        print()
