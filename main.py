from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model  = OllamaLLM(model="llama3.2:1b", temperature=0.1)

template = """
You are an expert for answering questions about a pizza restaurant.

Here are the relevant reviews: {reviews}

Here is the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


while True:
    print("------")
    question = str(input("Ask Your Question (q to quit):  "))
    print("------")
    if question == "q":
        break
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
    
