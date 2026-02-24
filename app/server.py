import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.CSVSession import CSVSession
from app.Inputs_models import Messages
from app.Upload_files import ParquetUploader

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

@app.post("/parquet/upload_file")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    try:
        uploader = ParquetUploader()
        return await uploader.upload_file_parquet(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.post("/askcsv/double/{process_id}")
async def ask_csv_with_tools(process_id: str, messages: Messages) -> JSONResponse:
    try:
        session = CSVSession(process_id)
        return await session.process_query(messages)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
