from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)
# temperture controls the randomness of a language model's output.
# higher values: more random, creative and diverse
# lower values: more deterministic and predictable

result = model.invoke("What is the captical of India?")

print(result)