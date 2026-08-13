from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"project": "MediScribeAI", "status": "running"}


@app.get("/health")
def read_health():
    return {"status": "healthy"}
