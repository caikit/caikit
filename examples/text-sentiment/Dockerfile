FROM registry.access.redhat.com/ubi9/python-39

RUN pip install -U pip setuptools wheel

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

#EXPOSE 7860

CMD python start_runtime.py
