from nano_graphrag import GraphRAG, QueryParam
import os
from dotenv import load_dotenv
dotenv_path = '.env'

load_dotenv(dotenv_path)
OPENAI_API_KEY = os.getenv('LLM_API_KEY')

graph_func = GraphRAG(working_dir="./test")

with open("./1.txt") as f:
    graph_func.insert(f.read())

# Perform global graphrag search
print(graph_func.query("星环科技2024年业绩怎么样"))

# Perform local graphrag search (I think is better and more scalable one)
print(graph_func.query("星环科技2024年业绩怎么样", param=QueryParam(mode="local")))
