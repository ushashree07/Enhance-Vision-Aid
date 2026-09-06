"""
Research Agent Module
====================
Autonomous AI agent for research and article writing using Groq LLM.
Searches web, extracts content, and generates news-worthy articles.
"""

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k
from voiceAss import speak
from dotenv import load_dotenv
import os

load_dotenv()


agent = Agent(
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGo(), Newspaper4k()],
    description="You are a senior NYT researcher writing an article on a topic.",
    instructions=[
        "For a given topic, search for the top 5 links.",
        "Then read each URL and extract the article text, if a URL isn't available, ignore it.",
        "Your response should be under 20 words."
        "Analyse and prepare an NYT worthy article based on the information."
    ],
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)

res = agent.run("Latest AI trends", stream=False)
s = str(res.content)
speak(s)

