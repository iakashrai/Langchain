from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.runnables import ConfigurableFieldSpec, RunnableWithMessageHistory

load_dotenv()

def get_chat_history(user_id, conversation_id):
    return SQLChatMessageHistory(
        table_name="chat_history",
        session_id=f"{user_id}_{conversation_id}",
        connection="sqlite:///sqlite.db"
    )


parser = CommaSeparatedListOutputParser()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant.
Keep your response in the following format:
{format_instructions}"""
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{msg}")
]).partial(
    format_instructions=parser.get_format_instructions()
)


model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.1,
    max_retries=2
)

# Keep model output as a string so RunnableWithMessageHistory
# can store the AI message correctly.
chain = prompt | model

config_fields = [
    ConfigurableFieldSpec(
        id="user_id",
        annotation=str,
        name="User ID",
        description="Unique identifier for a user.",
        default="",
        is_shared=True,
    ),
    ConfigurableFieldSpec(
        id="conversation_id",
        annotation=str,
        name="Conversation ID",
        description="Unique identifier for a conversation.",
        default="",
        is_shared=True,
    ),
]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="msg",
    history_messages_key="chat_history",
    history_factory_config=config_fields,
)


while True:

    user_input = input("You: ")

    if user_input == "000":
        break

    response = chain_with_history.invoke(
        {"msg": user_input},
        config={
            "configurable": {
                "user_id": "user1",
                "conversation_id": "conversation1"
            }
        }
    )

    parsed_response = parser.invoke(response)

    print("AI:", parsed_response)