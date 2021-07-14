FROM studyfang/hgn:latest

RUN python -m venv /opt/predictions_venv

COPY requirements.txt ./
RUN . /opt/predictions_venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt




