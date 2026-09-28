from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a summarization specialist AI assistant. Your mission is to summarize conversations using key points.",
        ),
        MessagesPlaceholder(variable_name="conversation"),
        ("human", "Summarize the conversation so far in {word_count} words."),
    ]
)

formatted_chat_prompt = chat_prompt.format(
    word_count=5,
    conversation=[
        ("human", "Hello! I’m Teddy. Nice to meet you."),
        ("ai", "Nice to meet you! I look forward to working with you."),
    ],
)

print(formatted_chat_prompt)