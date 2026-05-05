FROM python:3.11-slim
WORKDIR /app
COPY src/evaluation/requirements.txt .
RUN pip install -r requirements.txt
COPY src/evaluation .
CMD ["python", "run_evaluation.py"]