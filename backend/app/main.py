from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import router as auth_router
from app.consultations import router as consultations_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(consultations_router)


@app.get("/")
def read_root():
    return {"project": "MediScribeAI", "status": "running"}


@app.get("/health")
def read_health():
    return {"status": "healthy"}
