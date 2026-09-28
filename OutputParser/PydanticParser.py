from pydantic import BaseModel,Field
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate,MessagesPlaceholder, PromptTemplate


load_dotenv()

chat_history = [
    (
        'system',
        'You are acting as a chatbot answering user questions'
    ),
    (
        'human',
        'Hi How are you?'
    ),
    (
        'ai',
        'I am good, how about you?, how can i help you today?'
    ),
    (
        'human',
        'I am doing great, i want to know the temperature of Ranchi, Jharkhand'
    ),
    (
        'ai',
        ' Ranchi, Jharkhand Temperature : 29* Celcius'
    )
]

class Response(BaseModel):
    column:str=Field(min_length=10)
    value:int=Field(gt=10,lt=250)
    unit:str=Field(min_length=10)

parser = PydanticOutputParser(pydantic_object=Response)

user_prompt = PromptTemplate(
    template="""What is the {topic1}, {topic2}?

{format_instructions}
""",
    input_variables=["topic1", "topic2"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

user_message = HumanMessagePromptTemplate(prompt=user_prompt)

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("chat_history"),
    user_message
]).partial(
    chat_history=chat_history
)

llm = ChatGoogleGenerativeAI(
    model='gemini-3.8-flash'
)

chain = prompt | llm | parser

response = chain.invoke({
    'topic1':'AQI',
    'topic2':'WaterLevel'
}
)


print(response)
# column='Air Quality Index (AQI)' value=85 unit='AQI rating scale'
