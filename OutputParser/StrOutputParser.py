from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'You are acting as a {role}, answering the client queries'
        ),
        MessagesPlaceholder('chat_history'),
        (
            'human',
            'What is the issue in my {vechile} {component}'
        )
    ]
).partial(role="Mechanic")

parser = StrOutputParser()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.1
)

chain = prompt | llm | parser

response = chain.invoke(
    {
        "vechile":"Bike",
        "component":"clutch",
        "chat_history":[]
    }
)

print(response)

# Hey there! I’d be glad to help you figure that out, but since I don't have the bike on my lift right now, you’ll have to be my eyes andears. 

# Clutches usually fail in a few specific ways. **Which of these sounds like what you're experiencing?**

# 1. **Clutch Slipping:** You twist the throttle, the engine revs high and gets loud, but the bike doesn't accelerate like it should. (Usually means worn friction plates, incorrect cable tension, or the wrong engine oil).
# 2. **Clutch Dragging / Sticking:** You pull the lever all the way in, but the bike still wants to creep forward, or it's almost impossible to shift into Neutral while stopped. (Usually a stretched/loose cable, warped plates, or notched clutch basket).
# 3. **Bad Lever Feel:** The lever feels completely limp/loose, or it’s extremely stiff and hard to pull, or it doesn't snap back into place. (Points to a frayed/dry cable, broken return spring, or hydraulic fluid issues).
# 4. **Strange Noises:** You hear a squeal, chatter, or rattling sound when you pull the lever in or let it out. (Could be a bad throw-out/release bearing).

# Also, do me a favor and let me know:
# * **What is the Year, Make, and Model of the bike?**
# * **Does it have a cable clutch or a hydraulic clutch?**

# Give me a rundown of what’s happening, and we’ll get it diagnosed!