# chatbot

## 准备工作 Preparation

在代码config处提供你的API密钥
Set your API key in the code `config`


```python
config = {
        # "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",  # 可以修改为推理模型，比如 "deepseek-reasoner" deepseek-chat
        "system_message": "You are a helpful assistant.",
        "cost": args.cost
    }
```

或者将密钥存储在环境变量中。在`.bashrc`文件中添加以下行：
Or store the key in an environment variable. Add the following line to your `.bashrc` file:

```bash
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## 用法 Usage

启动脚本
Start the script

```bash
python3 ds.py
```

或者将别名添加到`.bashrc`文件中：
Or add an alias to your `.bashrc` file, `alias ds="python3 /home/zhicun/code/chatbot/ds.py"`
don't forget to `source ~/.bashrc`

```bash
zhicun@ZEN:~/code/chatbot$ ds
💬:
```

该脚本可以打印该次会话消耗的token数和成本明细，使用命令行参数`-c`或`--cost`来启用成本计算
The script can print the token count and cost details for the current session, using the command line argument `-c` or `--cost` to enable cost calculation

```bash
zhicun@ZEN:~/code/chatbot$ ds -c
💬: 你能告诉我你是谁吗?
🤖: 我是一个AI助手，旨在帮助你解决问题和提供信息.`

===== TOKEN 消耗明细 =====
输入: 4360 tokens [缓存命中: 4288 | 未命中: 72]
输出: 135 tokens
总消耗: 4495 tokens

===== 成本明细 =====
输入成本: ￥0.0023 元
输出成本: ￥0.0011 元
预估总成本: ￥0.0034 元
```


该脚本支持读取多个本地文件
The script supports reading multiple local files

```bash
zhicun@ZEN:~/code/chatbot$ ds
💬: 检查这些文件
    @file(README.md)
    @file(ds.py)
Reading file README.md...
Reading file ds.py...
🤖: # 文件检查报告
...
```

允许直接输入或者粘贴输入，输入完成后使用`ESC+ENTER`结束输入
Allow direct input or paste input, and use `ESC+ENTER` to end the input after completion
