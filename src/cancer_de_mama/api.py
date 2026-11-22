from functools import lru_cache
from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from src.cancer_de_mama.prediction.breast_cancer_predictor import BreastCancerPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "results" / "breast_cancer_model.joblib"
DATASET_PATH = PROJECT_ROOT / "data" / "breast_cancer.csv"
WEB_PAGE = Path(__file__).parent / "web" / "index.html"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ROWS = 2000
IGNORED_COLUMNS = {"id", "diagnosis"}

app = FastAPI(
    title="Classificador experimental de câncer de mama",
    description=(
        "Demonstração acadêmica para classificar registros de CSV. "
        "Não é uma ferramenta de diagnóstico."
    ),
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_predictor():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Modelo não encontrado. Execute `python main.py` antes de iniciar a aplicação."
        )
    return BreastCancerPredictor(MODEL_PATH)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB_PAGE)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.is_file()}


@app.get("/api/template", include_in_schema=False)
def download_template():
    if not DATASET_PATH.is_file():
        raise HTTPException(status_code=503, detail="Dataset de exemplo não encontrado.")

    dataframe = pd.read_csv(DATASET_PATH)
    if "diagnosis" not in dataframe.columns:
        raise HTTPException(status_code=500, detail="A coluna diagnosis não está no dataset.")

    sample = pd.concat(
        [
            dataframe[dataframe["diagnosis"] == "B"].head(1),
            dataframe[dataframe["diagnosis"] == "M"].head(1),
        ],
        ignore_index=True,
    )
    sample = sample.drop(
        columns=[
            column for column in sample.columns
            if column.casefold() in IGNORED_COLUMNS
            or column.casefold().startswith("unnamed:")
        ],
        errors="ignore",
    )
    if sample.empty:
        raise HTTPException(status_code=500, detail="Não foi possível gerar a planilha de exemplo.")

    content = sample.to_csv(index=False).encode("utf-8-sig")
    return StreamingResponse(
        BytesIO(content),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="modelo_medidas.csv"'},
    )


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Envie um arquivo com extensão .csv.")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="O CSV excede o limite de 5 MB.")
    if not contents:
        raise HTTPException(status_code=400, detail="O arquivo está vazio.")

    try:
        dataframe = pd.read_csv(BytesIO(contents), encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível ler o arquivo. Envie um CSV válido separado por vírgulas.",
        ) from error

    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    normalized_columns = [column.casefold() for column in dataframe.columns]
    if len(normalized_columns) != len(set(normalized_columns)):
        raise HTTPException(status_code=422, detail="O CSV contém nomes de coluna duplicados.")
    if dataframe.empty:
        raise HTTPException(status_code=400, detail="O CSV não contém registros.")
    if len(dataframe) > MAX_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"O CSV pode conter no máximo {MAX_ROWS} registros.",
        )

    metadata_columns = [
        column for column in dataframe.columns
        if column.casefold() in IGNORED_COLUMNS
        or column.casefold().startswith("unnamed:")
    ]
    id_column = next(
        (column for column in dataframe.columns if column.casefold() == "id"),
        None,
    )
    row_names = (
        dataframe[id_column].fillna("").astype(str).tolist()
        if id_column
        else [str(index + 1) for index in range(len(dataframe))]
    )
    features_frame = dataframe.drop(columns=metadata_columns)

    try:
        predictor = get_predictor()
        results = []
        for index, (_, row) in enumerate(features_frame.iterrows()):
            prediction = predictor.predict_one(row.to_dict())
            if prediction["score_type"] != "probability":
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "O modelo exportado não fornece probabilidades estimadas. "
                        "Execute `python main.py` novamente para atualizar o modelo."
                    ),
                )
            probability_malignant = prediction["score"]
            results.append(
                {
                    "row": row_names[index] or str(index + 1),
                    "diagnosis": prediction["diagnosis"],
                    "probability_benign": 1 - probability_malignant,
                    "probability_malignant": probability_malignant,
                }
            )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    benign_count = sum(result["diagnosis"] == "Benigno" for result in results)
    return {
        "total": len(results),
        "model": predictor.model_name,
        "summary": {
            "benign": benign_count,
            "malignant": len(results) - benign_count,
        },
        "results": results,
    }
