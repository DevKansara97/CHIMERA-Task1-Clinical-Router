FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

COPY requirements.txt /opt/app/
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user src/ /opt/app/src/
COPY --chown=user:user data/ /opt/app/data/
COPY --chown=user:user inference.py /opt/app/

USER user

ENTRYPOINT ["python", "inference.py"]
