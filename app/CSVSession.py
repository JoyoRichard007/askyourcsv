import os
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool
import pyarrow.parquet as pq
from langchain_community.vectorstores import FAISS
from cachetools import TTLCache
from dotenv import load_dotenv
from app.Inputs_models import PythonInputs, Messages

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

class CSVSession:
    def __init__(self, process_id: str):
        self.process_id = process_id
        self.message_history = ChatMessageHistory()
        self._cache = TTLCache(maxsize=100, ttl=1200)  # Cache avec expiration de 1 heure (3600 secondes)
        self.db = self.load_faiss_db()
        self.df = self.load_dataframe()
        self.store = {}
        self.agent_executor = self.create_agent_executor()

    def load_faiss_db(self):
        if 'faiss' in self._cache:
            return self._cache['faiss']
        embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
        db_path = os.path.join("vectorestore", self.process_id)
        faiss_db = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
        self._cache['faiss'] = faiss_db
        return faiss_db

    def load_dataframe(self):
        if 'df' in self._cache:
            return self._cache['df']
        file_path = os.path.join("parquet", f"{self.process_id}.parquet")
        parquet_table = pq.read_table(file_path)
        df = parquet_table.to_pandas()
        self._cache['df'] = df
        return df

    def create_agent_executor(self):
        TEMPLATE = """You are working with a pandas dataframe in Python. The dataframe name is `df`.
            It's important to understand the attributes of the dataframe before working with it. Here's the result of running `df.head(0).to_markdown()`:
            <df>
            {dhead}
            </df>
            These lines provide the shape and schema of the dataframe but are not meant to be used alone to answer questions.

            To ensure precise and relevant results, follow this iterative process:

            1. **Understand the Request**: Begin by fully understanding the user's question. If the request involves ambiguous column names or unclear details, ask the user to specify the relevant columns and provide additional information about the dataframe for better precision.

            2. **First Iteration - Explore with `data_search`**: Use `data_search` to explore the dataframe and identify the available columns and values. Document these findings thoroughly, as they will guide subsequent operations. `data_search` provides the foundational understanding that all further steps will be based on.

            3. **Second Iteration - Refine with `python_repl`**: Use `python_repl` to perform read-only operations strictly based on the validated columns and values obtained from `data_search`. Ensure that your operations, such as filtering, sorting, and grouping, are directly aligned with the dataframe's structure as identified by `data_search`.

            4. **Iterative Refinement - Review, Adjust, and Adapt**: After running the initial queries with `python_repl`, review the results. If they are not satisfactory or if errors persist, avoid redundant attempts. Instead, revisit the findings from `data_search` to adjust your approach or try a different strategy. This cycle of reviewing, adapting, and refining should continue until accurate and relevant results are achieved. If, after five iterations, the requested data cannot be found, stop the process and ask the user for additional information or clarification on why the request cannot be fulfilled.

            5. **Tool Synchronization and Chat History**: Both `data_search` and `python_repl` operate on the same dataframe `df`. Be aware that these tools are synchronized, and feel free to use `data_search` to validate and refine the queries you execute with `python_repl`. Additionally, use `chat_history` to reference previous conversations, but remember that the iterative process remains the same. This ensures full consistency in your operations.

            6. **Handle Errors Safely**: If you encounter errors such as "invalid syntax (<unknown>, line 1)" with `python_repl`, stop the iteration and inform the user that the request cannot be fulfilled due to potential safety concerns.

            Tips for success:
            * Always start with `data_search` to establish a clear understanding of the dataframe's structure.
            * Base all `python_repl` operations on `data_search` results, avoiding assumptions or operations not grounded in the data.
            * Follow the iterative process: 1. **Data Search**: Identify relevant data and structure. 2. **Python REPL**: Execute operations based on `data_search` results. 3. **Review and Adapt**: Refine, revalidate, or explore alternative approaches as necessary.
            * Ensure that each iteration builds on the previous one, maintaining coherence and accuracy.
            * Focus on providing high-quality answers by effectively using both tools in conjunction and ensuring that all queries and results are based on the context provided by `data_search`.
            * Use `chat_history` to provide context and continuity, but adhere to the iterative process for accuracy and precision.
            """

        retriever_tool = create_retriever_tool(self.db.as_retriever(), "data_search", "Search for relevant records")
        repl = PythonAstREPLTool(
            locals={"df": self.df},
            name="python_repl",
            description="Runs code to query the dataframe `df` only",
            args_schema=PythonInputs,
        )

        template = TEMPLATE.format(dhead=self.df.head().to_markdown())
        tools = [retriever_tool, repl]

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
        llm_with_tools = llm.bind_tools(tools)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )

        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
                "chat_history": lambda x: x["chat_history"]
            }
            | prompt | llm_with_tools | OpenAIToolsAgentOutputParser()
        )

        return AgentExecutor(agent=agent, tools=tools, verbose=True)

    async def process_query(self, messages: Messages) -> JSONResponse:
        try:
            for message in messages.messages:
                self.message_history.add_message({"role": message.role, "content": message.content})

        #     response = self.agent_executor.invoke(
        #         {"input": messages.messages[-1].content, "chat_history": self.message_history.messages}
        #     )

        #     return JSONResponse(content=dict(messages=[
        #         {"role": "assistant", "content": response['output']}
        #     ]), media_type="application/json", status_code=200)

        # except Exception as e:
        #     raise HTTPException(status_code=500, detail=f"Error: {e}")
            last_k_messages = 4

            def get_session_history(session_id):
                if session_id not in self.store:
                    self.store[session_id] = self.message_history

                stored_messages = self.store[session_id].messages

                if len(stored_messages) >= last_k_messages:
                    self.store[session_id].clear()
                
                for message in stored_messages[-last_k_messages:]:
                    self.store[session_id].add_message(message)
                
                return self.store[session_id]
            
            chain_with_message_history = RunnableWithMessageHistory(
                self.agent_executor,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            response = chain_with_message_history.invoke(
                {"input": messages.messages[-1].content},
                {"configurable": {"session_id": self.process_id}},
            )

            return JSONResponse(content=dict(messages=[
                {"role": "assistant", "content": response['output']}
            ]), media_type="application/json", status_code=200)
    
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {e}")