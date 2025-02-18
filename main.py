from fastapi import FastAPI
import os 
import uvicorn
import logging
import configparser
from fastapi.staticfiles import StaticFiles
from time import sleep
import sys
from change_ip import main as change_ip_main
from routers.model import MyHTTPException, my_exception_handler
from routers import infer as infer_router

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='fastapi.log', filemode='w')
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read(os.path.join('config','config.ini'))
host_ip = config['DEFAULT']['host'] 
port_num = config['DEFAULT']['port'] 
production = config['DEFAULT']['production']

script_name = "main"

app = FastAPI()
app.include_router(infer_router.router)

app.add_exception_handler(MyHTTPException, my_exception_handler)
os.makedirs('static', exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

def empty_to_false(value):
    value = value.strip('\n').strip()
    if value in (None, '', 'false', 'False', '0'):
        return False
    elif value in ('true', 'True', '1'):
        return True
        
    return bool(value)

# Endpoint to receive an image and start the processing pipeline
@app.get("/")
async def root():
    return {"detail":{"message": "Hello World"}}


def main():
    # change_ip_main()
    sleep(2)
    print('INITIALIZING FASTAPI SERVER')
    if empty_to_false(production) == False: 
        uvicorn.run(f"{script_name}:app", host=host_ip, port=int(port_num), reload=True, workers=1)
    else: uvicorn.run(f"{script_name}:app", host=host_ip, port=int(port_num), reload=False, workers=1)

if __name__ == "__main__":
    main()
