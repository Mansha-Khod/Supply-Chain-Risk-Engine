from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routers import overview,predict,explainability,segments

app=FastAPI(title='Supply Chain Risk Engine API',version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

app.include_router(overview.router,prefix="/api/overview",tags=['overview'])
app.include_router(predict.router,prefix="/api/predict",tags=['predict'])
app.include_router(explainability.router,prefix='/api/explainability',tags=['explainability'])
app.include_router(segments.router,prefix="/api/segments",tags=['segments'])

app.mount('/assets',StaticFiles(directory="reports/figures"),name='assets')
app.mount("/static",StaticFiles(directory="app/static"),name='static')

@app.get("/",include_in_schema=False)
def serve_frontent():
    return FileResponse("app/static/index.html")

