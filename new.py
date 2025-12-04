"""
简单示例：用 LangChain + SerpAPI 回答问题（手动输入 API Key 和模型）。

依赖：
    pip install -U langchain langchain-community langchain-openai serpapi

运行：
    python new.py
然后按提示输入 OPENAI_API_KEY、SERPAPI_API_KEY，可选输入模型名（默认 gpt-3.5-turbo）。
"""

import os
from getpass import getpass

from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI



def build_agent():
    openai_api_key = 'sk-4883291124d84711919484d5a6398beb'
    serpapi_api_key = 'e449f0248cb6dcae4c46049245652895ed96b887b04a6da600f6ba1a11235187'
    model = (os.environ.get("OPENAI_MODEL") or "gpt-3.5-turbo").strip()

    llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key='sk-4883291124d84711919484d5a6398beb',          # DeepSeek 的 API key
    openai_api_base="https://api.deepseek.com/v1",   # 必须加 /v1
    temperature=0.3
)
    # load_tools 可直接传 serpapi_api_key，llm 提供给工具链
    tools = load_tools(["serpapi"], llm=llm, serpapi_api_key=serpapi_api_key)

    return initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )


def main():
    agent = build_agent()
    question = "当前北京的温度是多少华氏度，这个温度的1/4是多少？"
    agent.run(question)


if __name__ == "__main__":
    main()
