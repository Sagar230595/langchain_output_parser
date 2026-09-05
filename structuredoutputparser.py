from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [ ]

for i in range(1,6):
    schema.append(ResponseSchema(name='fact_'+str(i), description='Fact '+str(i)+' about the topic'))
    # ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    # ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
    # ResponseSchema(name='fact_4', description='Fact 4 about the topic')


parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 5 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)