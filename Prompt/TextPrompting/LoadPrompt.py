from langchain_core.prompts import PromptTemplate,load_prompt

prompt_string = load_prompt("./Prompt/text prompting/prompt2.yaml", encoding="UTF-8")

print(prompt_string.format_prompt(country="India"))