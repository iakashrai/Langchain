from jinja2 import Environment,FileSystemLoader
from langchain_core.prompts import PromptTemplate

env = Environment(loader=FileSystemLoader("./Prompt/Jinja Prompting/"))
template = env.get_template("BasePrompt.txt")
jinja_formated_string = template.render({"user_name":"Akash","user_is_technical":"false"})
print(jinja_formated_string)

jinja_prompt = PromptTemplate(template=jinja_formated_string)

# Load the environment
extended_template = Environment(loader=FileSystemLoader("./Prompt/Jinja Prompting/"))
template = extended_template.get_template("ExtendedPrompt.txt")

# Phase 1: Jinja resolves 'user_name' and 'user_is_technical'. 
# The {% raw %} block tells it to leave the 'programming_language' loop intact as a string.
jinja_formated_string = template.render({
    "user_name": "Akash",
    "user_is_technical": "True"
})

# Phase 2: LangChain receives a template string that still contains the Jinja for loop.
jinja_prompt = PromptTemplate(
    template=jinja_formated_string,
    template_format="jinja2",
    input_variables=["programming_language"]
)

# LangChain executes the final render, filling in the programming languages.
print(jinja_prompt.format(programming_language=["Java", "C++", "Python"]))