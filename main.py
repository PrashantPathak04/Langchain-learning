import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langsmith import traceable

load_dotenv()
os.environ.setdefault("LANGSMITH_PROJECT", "langchain-learning")

@traceable  # This decorator auto-traces the function
def main():
    # print("Hello, World!")
    information = "You are a helpful assistant that translates English to Hindi."
    template = "Given information: {information}, translate the following English text to Hindi: {text}"
    prompt_template = PromptTemplate(template=template, input_variables=["information", "text"])
    llm = OllamaLLM(model="qwen2.5:7b", base_url="http://localhost:11434", temperature=1)
    chain = prompt_template | llm
    response = chain.invoke({"information": information, "text": "Hello, how are you?"})
    print(response)

if __name__ == "__main__":
    main()


