FROM python:3.12-slim

LABEL org.opencontainers.image.authors="fengwang@gzhu.edu.cn"

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home notebook

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /tmp/requirements.txt

WORKDIR /notebooks
COPY --chown=notebook:notebook . /notebooks

EXPOSE 8888

USER notebook

CMD ["python3", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/notebooks"]
