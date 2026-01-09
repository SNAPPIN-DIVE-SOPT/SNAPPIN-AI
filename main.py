import os
from io import BytesIO
from urllib.parse import unquote_plus

import boto3
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('clip-ViT-B-32', cache_folder='./model_cache')
s3 = boto3.client('s3', region_name='ap-northeast-2')
SPRING_SERVER_URL = os.environ.get('SPRING_SERVER_URL')


def handler(event, context):
    for record in event['Records']:
        bucket_name = record['s3']['bucket']['name']
        raw_key = record['s3']['object']['key']
        key = unquote_plus(raw_key)

        try:
            print(f"▶▶▶ [DEBUG] Bucket: {bucket_name} / Key: {key}")
            # 1. S3에서 이미지 가져오기
            file_obj = s3.get_object(Bucket=bucket_name, Key=key)
            img = Image.open(BytesIO(file_obj['Body'].read()))

            # 2. 이미지 -> 벡터 변환
            embedding = model.encode(img).tolist()

            # 3. 스프링부트로 전송 (순수 벡터만 보냄)
            payload = {
                "imageUrl": key,
                "embedding": embedding
            }

            requests.post(
                f"{SPRING_SERVER_URL}/api/v1/photos/process",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

        except Exception as e:
            print(f"Error: {e}")
            raise e

    return {"statusCode": 200, "body": "OK"}
