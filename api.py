# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from model_handler import predict_sms


app = FastAPI(title="SMS Spam Detection API")


class Message(BaseModel):
text: str


@app.get("/")
def root():
return {"message": "SMS Spam Detection API. POST /predict with {text}."}


@app.post("/predict")
def predict(message: Message):
res = predict_sms(message.text)
return res
