from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uuid
import os
import csv
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")

app = FastAPI()

# 🔥 Chemins ABSOLUS pour Docker / Coolify
BASE_PATH = "/app"
TMP_DIR = os.path.join(BASE_PATH, "tmp")
PARQUET_DIR = os.path.join(BASE_PATH, "parquet")
VECTORSTORE_DIR = os.path.join(BASE_PATH, "vectorestore")

# Création sécurisée des dossiers
for directory in [TMP_DIR, PARQUET_DIR, VECTORSTORE_DIR]:
    os.makedirs(directory, exist_ok=True)


class ParquetUploader:
    def __init__(self, embeddings_model: str = "text-embedding-3-small"):
        self.embeddings_model = embeddings_model

    async def verify_csv_extension(self, file: UploadFile) -> None:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    async def create_process_id(self) -> str:
        return str(uuid.uuid4())

    async def save_temp_file(self, file: UploadFile):
        content = await file.read()

        with NamedTemporaryFile(dir=TMP_DIR, delete=False, suffix=".csv") as temp_file:
            temp_file.write(content)
            file_path = temp_file.name

        os.chmod(file_path, 0o666)
        return file_path, content

    async def detect_separator(self, content: bytes) -> str:
        decoded_content = content.decode("latin1")
        dialect = csv.Sniffer().sniff(decoded_content)
        return dialect.delimiter

    async def load_csv_with_pandas(self, file_path: str, separator: str):
        # 🔥 On force l'encodage (solution au bug Coolify)
        df = pd.read_csv(file_path, sep=separator, encoding="latin1")

        # Conversion en Documents LangChain
        documents = [
            Document(page_content=row.to_string())
            for _, row in df.iterrows()
        ]

        return documents, df

    async def save_to_faiss(self, documents, process_id: str):
        embedding = OpenAIEmbeddings(model=self.embeddings_model)
        db = FAISS.from_documents(documents, embedding)

        db_path = os.path.join(VECTORSTORE_DIR, process_id)
        db.save_local(db_path)

    async def save_to_parquet(self, df: pd.DataFrame, process_id: str):
        parquet_path = os.path.join(PARQUET_DIR, f"{process_id}.parquet")

        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path)

    async def clean_up(self, file_path: str):
        if os.path.exists(file_path):
            os.remove(file_path)

    async def upload_file_parquet(self, file: UploadFile) -> JSONResponse:
        try:
            await self.verify_csv_extension(file)

            process_id = await self.create_process_id()

            file_path, content = await self.save_temp_file(file)
            print("DEBUG: File saved ->", file_path)

            separator = await self.detect_separator(content)
            print("DEBUG: Separator detected ->", separator)

            documents, df = await self.load_csv_with_pandas(file_path, separator)
            print("DEBUG: CSV loaded, rows ->", len(df))

            await self.save_to_faiss(documents, process_id)
            print("DEBUG: Saved to FAISS")

            await self.save_to_parquet(df, process_id)
            print("DEBUG: Saved to Parquet")

            await self.clean_up(file_path)
            print("DEBUG: Temp file removed")

            return JSONResponse(
                content={
                    "message": "File uploaded successfully",
                    "file_id": process_id,
                    "file_name": file.filename,
                    "separator": separator
                },
                status_code=200
            )

        except Exception as e:
            print("ERROR:", str(e))
            return JSONResponse(
                content={"message": f"An error occurred: {e}"},
                status_code=500
            )


uploader = ParquetUploader()


@app.post("/parquet/upload_file")
async def upload_file(file: UploadFile):
    return await uploader.upload_file_parquet(file)