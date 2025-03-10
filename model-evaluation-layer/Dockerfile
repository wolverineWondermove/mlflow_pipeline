FROM python:3.10
RUN pip install -U uv
WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
COPY evaluate.py .
EXPOSE 8000
ENTRYPOINT ["uvicorn","evaluate:app","--host","0.0.0.0","--port","8000"]