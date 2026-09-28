from time import time
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from dotenv import load_dotenv

load_dotenv()

# set_llm_cache(InMemoryCache())
# Time: 2.68 seconds
# **Redis** is an open-source, ultra-fast, in-memory data store commonly used as a database, cache, and message broker.
# --------------------------------------------------
# Time: 0.00 seconds
# **Redis** is an open-source, ultra-fast, in-memory data store commonly used as a database, cache, and message broker.
# --------------------------------------------------

# When i changed the question little bit
# call_llm("Explain Redis in one sentence.")
#
# call_llm("Tell me about Redis in one sentence.")
# Time: 10.37 seconds
# **Redis** is an ultra-fast, in-memory data store used primarily as a high-performance database, cache, and message broker.
# --------------------------------------------------
# Time: 2.68 seconds
# Redis is an open-source, in-memory data structure store commonly used as an ultra-fast database, cache, and message broker.
# --------------------------------------------------

# --------------------------------------------------
# -------------Redis Caching------------------------
# redis_client = redis.Redis(
#     host="localhost",
#     port=6379,
#     decode_responses=False
# )

# set_llm_cache(
#     RedisCache(
#         redis_client,
#         ttl=3600
#     )
# )
# --------------------------------------------------


# Semnatic Caching for similar type of quesitons


llm = GoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def call_llm(question):
    start = time()

    response = llm.invoke(question)

    elapsed = time() - start

    print(f"Time: {elapsed:.2f} seconds")
    print(response)
    print("-" * 50)


# First call -> CACHE MISS
call_llm("Explain Redis in one sentence.")

# Same input -> CACHE HIT
call_llm("Tell me about Redis in one sentence.")
