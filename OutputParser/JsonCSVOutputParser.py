from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser,CommaSeparatedListOutputParser
from dotenv import load_dotenv

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

# parser = JsonOutputParser()
# '''
# {'location': 'Ranchi, Jharkhand', 'aqi': 88, 'category': 'Moderate', 'primary_pollutant': 'PM2.5', 'status': 'Air quality is acceptable; however, sensitive individuals may experience minor irritation.'}
# '''

# parser = CommaSeparatedListOutputParser()
# ['95', 'Moderate', 'PM2.5']

parser = BaseCumulativeTransformOutputParser()


user_prompt = PromptTemplate(
    template = '''Also tell me about the current {topic}
    
    {format_instructions}
    ''',
    input_variables=['topic'],
    partial_variables={
        "format_instructions":parser.get_format_instructions()
    }
)

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate(prompt=user_prompt)
    ]
).partial(chat_history=chat_history)

llm = ChatGoogleGenerativeAI(
    model='gemini-3.8-flash'
)

chain = prompt | llm | parser

response = chain.invoke({'topic':'AQI'})

print(response)