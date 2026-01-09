FROM public.ecr.aws/lambda/python:3.10

RUN yum update -y && \
    yum install -y gcc gcc-c++ git

RUN pip install --upgrade pip

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('clip-ViT-B-32', cache_folder='./model_cache')"

COPY main.py ${LAMBDA_TASK_ROOT}

CMD [ "main.handler" ]