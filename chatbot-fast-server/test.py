import requests

def get_res(index):
    r = requests.post(url='https://hf.space/embed/nitishkumargundapu793/chat-bot-response/api/predict/',json={'fn_index': 2, 'data': [index], 'session_hash': "73g8yepkp3b"})
    return r.json()['data'][1]