from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_chroma.vectorstores import Chroma

examples = [
    {
        "question": "Who lived longer, Steve Jobs or Einstein?",
        "answer": """Does this question require additional questions: Yes.
Additional Question: At what age did Steve Jobs die?
Intermediate Answer: Steve Jobs died at the age of 56.
Additional Question: At what age did Einstein die?
Intermediate Answer: Einstein died at the age of 76.
The final answer is: Einstein
""",
    },
    {
        "question": "When was the founder of Google born?",
        "answer": """Does this question require additional questions: Yes.
Additional Question: Who is the founder of Google?
Intermediate Answer: Google was founded by Larry Page and Sergey Brin.
Additional Question: When was Larry Page born?
Intermediate Answer: Larry Page was born on March 26, 1973.
Additional Question: When was Sergey Brin born?
Intermediate Answer: Sergey Brin was born on August 21, 1973.
The final answer is: Larry Page was born on March 26, 1973, and Sergey Brin was born on August 21, 1973.
""",
    },
    {
        "question": "Who was the President when Donald Trump's mother was born?",  # 쉼표 추가
        "answer": """Does this question require additional questions: Yes.
Additional Question: Who is Donald Trump's mother?
Intermediate Answer: Donald Trump's mother is Mary Anne MacLeod Trump.
Additional Question: When was Mary Anne MacLeod Trump born?
Intermediate Answer: Mary Anne MacLeod Trump was born on May 10, 1912.
Additional Question: Who was the U.S. President in 1912?
Intermediate Answer: William Howard Taft was President in 1912.
The final answer is: William Howard Taft
""",
    },
    {
        "question": "Are the directors of Oldboy and Parasite from the same country?",
        "answer": """Does this question require additional questions: Yes.
Additional Question: Who is the director of Oldboy?
Intermediate Answer: The director of Oldboy is Park Chan-wook.
Additional Question: Which country is Park Chan-wook from?
Intermediate Answer: Park Chan-wook is from South Korea.
Additional Question: Who is the director of Parasite?
Intermediate Answer: The director of Parasite is Bong Joon-ho.
Additional Question: Which country is Bong Joon-ho from?
Intermediate Answer: Bong Joon-ho is from South Korea.
The final answer is: Yes
""",
    },
]

embedding = FakeEmbeddings(size=120)
vector_db = Chroma(database="example_store",embedding_function=embedding)

question_text = [example["question"] for example in examples]
question_metadata = [example for example in examples]
vector_db.add_texts(question_text,question_metadata)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vector_db,
    k=2
)

example_prompt_template = PromptTemplate.from_template(
    "Question:\n{question}\nAnswer:\n{answer}\n"
)

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt_template,
    suffix="Question:\n{question}\nAnswer:",
    input_variables=["question"]
)

question = "When did Bill Gates founded Microsoft?"
final_prompt_string = prompt.format(question=question)
print(final_prompt_string)