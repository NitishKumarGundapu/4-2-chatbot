from fastapi import *
from application import *
from states import get_address
from pydantic import BaseModel
from googletrans import Translator
from fastapi.middleware.cors import CORSMiddleware

translator = Translator()
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/translater")
async def translater(msg='nice',lang='english'):
    result = translator.translate(msg,dest=lang)
    return str(result.text)


@app.get("/translation")
async def translation(msg='nice',lang='english'):
    res = []
    messages = msg.split(",")
    for a in messages:
        res.append(translator.translate(a,dest=lang).text)
    return res

@app.get("/address")
async def getAddress(pincode=500010,lang='english'):
    res = []
    result1 = translator.translate("The Nearest Store is : ",dest=lang).text
    messages = get_address(int(pincode))
    for a in messages:
        res.append(translator.translate(a,dest=lang).text)
    return result1 + res[0]

@app.get("/get_response")
async def getResponse(msg='nice',lang='english'):
    r = chatbot_response(msg)
    res = translator.translate(r,dest=lang).text
    return res



@app.get("/")
async def root():
    return {"message": "this is chatbot-response-server"}