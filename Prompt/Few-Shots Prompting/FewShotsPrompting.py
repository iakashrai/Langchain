from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate

example_prompt = PromptTemplate(
    template="Question:{question}\n Answer:{answer}"
)

examples = [
    {
        "question": "Who lived longer, Steve Jobs or Einstein?",
        "answer":"""Does this question require additional questions: Yes. 
                    Additional Question: At what age did Steve Jobs die?
                    Intermediate Answer: Steve Jobs died at the age of 56.
                    Additional Question: At what age did Einstein die?
                    Intermediate Answer: Einstein died at the age of 76.
                    The final answer is: Einstein
                """
    },
    {
        "question": "When was the founder of Naver born?",
        "answer":"""Does this question require additional questions: Yes.
                    Additional Question: Who is the founder of Naver?
                    Intermediate Answer: Naver was founded by Lee Hae-jin.
                    Additional Question: When was Lee Hae-jin born?
                    Intermediate Answer: Lee Hae-jin was born on June 22, 1967.
                    The final answer is: June 22, 1967
                """
    }
]

# print(example_prompt.format(**examples[1]))

few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    suffix="Question:\n{question}\nAnswer:",
    input_variables=["question"]
)
question = input("Enter your Question here:\n")
final_prompt = few_shot_prompt.format(question = question)
print(final_prompt)

