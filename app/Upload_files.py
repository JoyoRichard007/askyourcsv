from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uuid
import os
import csv
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io import StringIO
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

app = FastAPI()

tmp_dir = 'tmp'

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

class ParquetUploader:
    def __init__(self, embeddings_model: str = "text-embedding-ada-002"):
        self.embeddings_model = embeddings_model

    async def verify_csv_extension(self, file: UploadFile) -> None:
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Error: Only CSV files are accepted.")

    async def create_process_id(self) -> str:
        return str(uuid.uuid4())

    async def save_temp_file(self, file: UploadFile) -> str:
        with NamedTemporaryFile(dir = tmp_dir, delete=False, suffix=".csv") as temp_file:
            file_path = temp_file.name
            content = await file.read()
            temp_file.write(content)  # Pas besoin d'await ici
            # Ajout des permissions pour que FastAPI puisse lire/écrire
            os.chmod(file_path, 0o666)
        return file_path, content

    async def detect_separator(self, content: bytes) -> str:
        decoded_content = content.decode('latin1')
        dialect = csv.Sniffer().sniff(decoded_content)
        return dialect.delimiter

    async def load_csv(self, file_path: str, separator: str):
        loader = CSVLoader(file_path=file_path, csv_args={"delimiter": separator})
        return loader.load()

    async def save_to_faiss(self, data, process_id: str):
        embedding = OpenAIEmbeddings(model=self.embeddings_model)
        db = FAISS.from_documents(data, embedding)
        # db_path = f"./vectorestore/{process_id}"
        db_path = os.path.join("vectorestore", process_id)
        db.save_local(db_path)

    async def save_to_parquet(self, content: bytes, separator: str, process_id: str):
        df = pd.read_csv(StringIO(content.decode('latin1')), sep=separator)
        # parquet_path = f"./parquet/{process_id}.parquet"
        parquet_path = os.path.join("parquet", f"{process_id}.parquet")
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path)

    async def clean_up(self, file_path: str):
        os.remove(file_path)

    async def upload_file_parquet(self, file: UploadFile) -> JSONResponse:
        try:
            await self.verify_csv_extension(file)
            process_id = await self.create_process_id()
            file_path, content = await self.save_temp_file(file)

            # ===== DEBUG TEMPORAIRE =====
            from pathlib import Path
            tmp_file = Path(file_path)
            print("DEBUG: file_path =", tmp_file)
            print("DEBUG: Exists?", tmp_file.exists())
            print("DEBUG: Readable?", tmp_file.stat())
            # ============================

            separator = await self.detect_separator(content)
            data = await self.load_csv(file_path, separator)
            await self.save_to_faiss(data, process_id)
            await self.save_to_parquet(content, separator, process_id)
            await self.clean_up(file_path)

            return JSONResponse(
                content={
                    "message": "File uploaded successfully",
                    "file_id": process_id,
                    "file_name": file.filename,
                    "separator": separator
                },
                status_code=200
            )

        except (csv.Error, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            return JSONResponse(
                content={"message": f"CSV parsing error: {e}"},
                status_code=400
            )
        except Exception as e:
            return JSONResponse(
                content={"message": f"An error occurred: {e}"},
                status_code=500
            )