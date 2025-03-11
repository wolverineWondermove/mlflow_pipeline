FROM python:3.9-slim

# 시스템 패키지 업데이트 및 Git 설치
RUN apt-get update && apt-get install -y git

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Git 관련 경고를 숨기려면 필요 시 환경 변수 설정
ENV GIT_PYTHON_REFRESH=quiet

COPY app.py .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
