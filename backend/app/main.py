from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.audio import router as audio_router
from app.auth import router as auth_router
from app.consultations import router as consultations_router
from app.fog import router as fog_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(consultations_router)
app.include_router(audio_router)
app.include_router(fog_router)


@app.get("/")
def read_root():
    return {"project": "MediScribeAI", "status": "running"}


@app.get("/health")
def read_health():
    return {"status": "healthy"}
