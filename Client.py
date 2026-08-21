from langchain.agents import create_agent   

from langchain_mcp_adapters.client import MultiServerMCPClient 
import os 
from dotenv import load_dotenv 
load_dotenv()
from langchain_groq import ChatGroq 
key = os.getenv("GROQ_API_KEY")
chat_model = ChatGroq(model="openai/gpt-oss-120b")

async def main():
    client = MultiServerMCPClient({
        "edumind_mcp":{
            "command":"python",
            "args":["server.py"],
            "transport":"stdio"
        }
    })
    get_tools = await client.get_tools() 
    if not get_tools:
        print("tools cant find out")

    LLM_blind = chat_model.bind_tools(get_tools) 

    system_prompt = """You are a research assistant with access to tools.

    Rules, in order:
    1. For ANY question, call Retrival first to check the document.
    2. If Retrival returns relevant info, answer using it.
    3. If Retrival returns nothing useful, call web_search.
    4. For math, use calculator.

    Never answer from your own knowledge without calling a tool first."""

    agent = create_agent(model=LLM_blind, tools=get_tools, system_prompt=system_prompt)

    query = "what is the largest country in the world right now ?"
    response = await agent.ainvoke({'messages':[('user',query)]})
    print(response['messages'][-1].content)

import asyncio 

if __name__ == "__main__":
    asyncio.run(main())

