from fastapi import FastAPI, File, UploadFile,Request
import os 
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import Website.app.api.utils.helpermethodsWeb as helpermethodsWeb
from fastapi.responses import RedirectResponse
from typing import List

app = FastAPI()
templates = Jinja2Templates(directory="./Website/app/api/templates")
app.mount("/images", StaticFiles(directory="./Website/Temp"), name="images")


@app.get("/")
async def main(request : Request = None):
    return templates.TemplateResponse("index.html", context={"request":request})

@app.post("/upload")
async def upload_images(images: List[UploadFile] = File(...)):
    for image in images:
        contents = await image.read()
        path = f"./Website/Temp/{image.filename}"
        with open(path, "wb") as f:
            f.write(contents)
        f.close()
    return RedirectResponse(url="/images-grid", status_code=303)



@app.get("/predict")
async def predict():
    results = helpermethodsWeb.infer()
    return {"results": results}

@app.get("/images-grid")
async def get_image_grid(request : Request = None):
    path = f".\\Website\\Temp"
    image_paths = os.listdir(path)
    image_paths = ['images/' + path for path in image_paths]
    if len(image_paths) != 0:
        predict = helpermethodsWeb.infer()
    res = [dict(path=path, label=predict) for path, predict in zip(image_paths, predict)]
    context = {"request":request,"Infer_list": res}
    return templates.TemplateResponse("gallery.html", context=context)


@app.delete("/delete")
async def onload():
    path = f".\\Website\\Temp"
    helpermethodsWeb.empty_folder(path)
    return {"message": "Deleted all images"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
