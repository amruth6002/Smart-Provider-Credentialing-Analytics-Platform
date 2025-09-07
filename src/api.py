from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
from typing import Optional
from .engine import ProviderDQEngine
from .nlu import parse_intent

app = FastAPI(title="Provider DQ Engine API")
engine = ProviderDQEngine()

@app.post("/upload")
async def upload(
    roster: UploadFile = File(...),
    ny: Optional[UploadFile] = File(None),
    ca: Optional[UploadFile] = File(None),
    npi: Optional[UploadFile] = File(None),
):
    def save_temp(f: UploadFile) -> str:
        p = f"/tmp/{f.filename}"
        with open(p, "wb") as out:
            out.write(await f.read())  # type: ignore
        return p
    r_path = await save_temp(roster)
    ny_path = await save_temp(ny) if ny else None
    ca_path = await save_temp(ca) if ca else None
    npi_path = await save_temp(npi) if npi else None
    engine.load_files(r_path, ny_path, ca_path, npi_path)
    return {"status": "ok"}

@app.post("/query")
async def query(text: str = Form(...)):
    intent, params = parse_intent(text)
    res = engine.run_query(intent, params)
    if isinstance(res, pd.DataFrame):
        return JSONResponse({"intent": intent, "params": params, "rows": res.to_dict(orient="records")})
    return {"intent": intent, "params": params, "result": res}