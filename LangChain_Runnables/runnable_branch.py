from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# 1st prompt
prompt1 = PromptTemplate(
    template="Write a detailed report about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the following text {text}",
    input_variables=['text']
)


parser = StrOutputParser()

para_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x:len(x.split()) >= 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)


chain = RunnableSequence(para_gen_chain,branch_chain)


result = chain.invoke({'topic':'cricket'})

print(result)