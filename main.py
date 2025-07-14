from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from summarizer import summarize_text

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    summary = request.session.pop("summary", None)  # 한번 보여주고 삭제
    return templates.TemplateResponse("index.html", {"request": request, "summary": summary})

@app.post("/summarize")
async def summarize(request: Request, text: str = Form(...)):
    summary = summarize_text(text)
    request.session["summary"] = summary  # 세션에 저장
    return RedirectResponse(url="/", status_code=303)
