from src.model import suppaluk
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
from src.model import suppaluk
import io

model = suppaluk()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def index():
    return "Suppaluk"


@app.post('/classify')
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    img = model.readImgByte(contents)
    img_pred = model.predict(img)
    result = model.showImgBlob(img_pred)

    return StreamingResponse(io.BytesIO(result), media_type='image/jpg')
    
