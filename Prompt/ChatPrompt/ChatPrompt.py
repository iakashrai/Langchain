from langchain_core.prompts import ChatPromptTemplate

chatPrompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)

chatPrompt_string = chatPrompt.format_messages(name="ChatGPT",user_input="What is your name?")
print(chatPrompt_string)