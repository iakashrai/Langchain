from langchain_core.prompts import PromptTemplate
from datetime import time;

template = """
You are a {role}, Explain me on the {topic} Topic,
Keep the output in the below format.

1. Overview
2. Explanation
3. Conclusion
4. Summary
5. Reference 
"""

### Using from_template method.

prompt = PromptTemplate.from_template(template=template)
formated_prompt = prompt.format(role="Doctor",topic="Human DNA")
print(formated_prompt)

### Using PromptTemplate Class

prompt = PromptTemplate(template=template,input_variables=["role","topic"])
print(prompt.format(role="Cricket Expert",topic="Impact Player Rule"))

### Using partial_variables
# """
# Suppose we want to specify the current date in your prompt, 
# Hardcoding the date into the prompt or passing it along with other input variables may not be practical. \
# In this case, using a function that returns the current date to modify the prompt partially is much more convenient.
# """

def get_time():
    t = time.strftime
    return str(t)

template = "Hi Current time is {time}, list down top {n} significance of current time"

prompt = PromptTemplate(template=template,input_variables=["n"],partial_variables={"time":get_time})

print(prompt.format(n="5"))




