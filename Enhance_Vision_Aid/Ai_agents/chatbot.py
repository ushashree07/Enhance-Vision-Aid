"""
Chatbot Module
==============
Conversational AI using Groq LLM with memory management.
Provides both simple chat and stateful conversation processing.
"""

from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq API client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def chat_respond(message):
    """Simple chat response using Groq API."""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Give short answers.",
            },
            {
                "role": "user",
                "content": message,
            }
        ],
        model="llama3-8b-8192",  # Use a valid model
        temperature=0.5,
        max_completion_tokens=500,
        top_p=1,
        stop=None,
        stream=False,
    )

    # Print AI response
    res=str(chat_completion.choices[0].message.content)
    return res



import os
import time
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda

class LanguageModelProcessor:
    """
    Stateful language model processor with conversation memory.
    Maintains chat history and processes queries using Groq Mixtral model.
    """
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        # Use new LangChain pipeline syntax
        self.conversation = self.prompt | self.llm

    def process(self, text):
        """Process user input and return AI response with memory."""
        self.memory.chat_memory.add_user_message(text)
        start_time = time.time()

        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response.content)  # Store response in memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response.content}")
        return response.content
