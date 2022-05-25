FROM python:3.8.6-buster

COPY requirements.txt ./
COPY api ./api
COPY deep_seo ./deep_seo
COPY model.joblib ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT


# write some code to build your image
