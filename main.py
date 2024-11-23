from fastapi import FastAPI, Request, Path, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os


app = FastAPI()

# app.mount("/static", StaticFiles(directory="pkg"), name="pkg")
templates = Jinja2Templates(directory="templates")

# def get_response() -> str:
#     # with open("pkg")

#     return content


@app.get("/", response_class=HTMLResponse)
async def read_root(request:Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={}
    )

@app.get("/js/{filename}")
async def get_js(filename: str):
    # if filename != "rust_webgpu.js":
    #     return Response(status_code=403)
    
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "pkg", filename)
    
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            content = f.read()

        media_type = (
            "application/wasm" if filename.endswith(".wasm") else "application/javascript"
        )
        return Response(content, media_type=media_type)
    
    return Response(status_code=404)