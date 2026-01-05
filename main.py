import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os

img_model = SentenceTransformer('clip-ViT-B-32')
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

IMAGE_TAG_DEFINITIONS = {
    #  장면구성 태그
    "내추럴": "자연스러운 순간 포착처럼 보이는 이미지로, 인물이 카메라를 의식하지 않은 듯한 인상을 준다. 시선이 렌즈가 아닌 다른 곳을 향하고 있으며, 걷거나 움직이는 동작의 중간을 멈춘 장면처럼 느껴진다. 프레임이 완벽히 대칭적이지 않거나 여백이 많아, 연출보다는 관찰자의 시선으로 기록된 장면에 가깝다.",

    "연출된": "사진을 위해 인물의 포즈와 구도가 의도적으로 정돈된 이미지. 인물이 카메라를 또렷하게 인식하고 있으며, 몸과 얼굴의 방향이 안정적으로 구성되어 있다. 프레임 중심에 인물이 위치하는 경우가 많고, 전체적으로 세팅된 느낌이 강해 인위적인 연출이 명확히 드러난다.",

    "서사적": "인물과 장소가 함께 프레임에 포함되어 있으며, 특정 상황이나 사건이 진행 중인 장면이 명확하게 보이는 이미지. 의상, 색감, 소품, 배경이 하나의 콘셉트로 의도적으로 통일되어 있고, 인물이 완성된 포즈를 취하기보다 행동의 중간 단계에 놓여 있어 장면의 앞뒤 맥락이 자연스럽게 암시될 때 해당된다.",

    # 분위기 태그
    "따스한": "노란 기운의 색온도와 부드러운 그림자가 느껴지며, 차갑지 않고 포근한 인상을 주는 이미지. 노을, 오후 햇빛, 실내의 노란 조명처럼 따뜻한 빛이 주를 이루고 가을·겨울 사진에서도 온기가 느껴질 때 해당된다.",

    "청량한": "맑고 시원한 공기가 느껴지는 밝은 이미지로, 파란 하늘이나 초록 풍경이 또렷하게 보이고 색 대비가 명확하다. 여름 낮처럼 답답함 없이 개방감과 상쾌함이 강조될 때 적합하다.",

    "투명한": "색 보정이 과하지 않고 전체적으로 깨끗하고 가벼운 인상을 주는 이미지. 흰색이나 연한 색상이 탁하지 않게 표현되며 화면이 클린하고 정돈되어 보일 때 해당된다. 자연광에서 찍은 사진이 해당",

    "몽환적인": "초점이 또렷하지 않거나 대비가 낮아 현실감이 줄어든 이미지. 빛이 번지거나 흐릿하게 퍼지며, 실제 장면보다 꿈처럼 비현실적으로 느껴질 때 해당된다.",

    "뚜렷한": "윤곽과 경계가 선명하게 드러나 시선이 흐트러지지 않는 이미지. 인물과 배경의 구분이 명확하고, 색과 형태가 뭉개지지 않아 화면의 중심이 분명하게 느껴진다. 초점이 정확히 잡혀 있고 대비가 적절해 디테일이 또렷하게 인식될 때 해당된다.",

    "차가운": "푸른 기운이나 중성적인 색온도가 중심이 되어 따뜻한 색감이 거의 느껴지지 않는 이미지. 인공광이나 도시의 조명이 주광원으로 작용하며, 색 대비와 명암이 비교적 분명해 정돈되고 도시적인 인상을 준다. 노을이나 자연광의 온기보다는 밤의 가로등, 간판, 실내 조명처럼 차분하고 거리감 있는 빛이 강조될 때 해당된다.",

    # 스타일 태그
    "디지털": "노출과 화이트 밸런스가 안정적으로 맞춰져 있고, 노이즈가 거의 없음.  색 보정이 과하지 않아 실제 눈으로 보는 장면에 가까운 이미지. 윤곽이 또렷하고 질감보다 선명함과 정제된 표현이 강조될 때 해당된다.",

    "아날로그": "필름 사진처럼 색 번짐과 부드러운 대비, 미세한 질감이 느껴지는 이미지. 노이즈나 노출의 불완전함이 자연스럽게 드러나며, 디지털처럼 정확하기보다 감각적으로 기록된 인상이 강할 때 해당된다.",

    "Y2K": "플래시 사용이나 높은 노출로 화면이 밝게 뜨며, 색 대비가 강하고 피부톤이 하얗게 표현되는 이미지. 자연광보다 인공광의 존재감이 크고, 즉흥적이고 키치한 인상이 강조될 때 해당된다."
}

TAG_LABELS=list(IMAGE_TAG_DEFINITIONS.keys())
TAG_TEXTS=list(IMAGE_TAG_DEFINITIONS.values())


def analyze_images_individually(image_paths):
    tag_embeddings = text_model.encode(TAG_TEXTS)
    final_results = []

    print(f"총 {len(image_paths)}장의 사진 개별 분석 시작 (상세 설명 기반)...\n")

    for path in image_paths:
        try:
            if not os.path.exists(path):
                print(f"파일 없음: {path}")
                continue

            img = Image.open(path)
            img_emb = img_model.encode(img)

            scores = util.cos_sim(img_emb, tag_embeddings)[0].numpy()

            mapped_scores = []
            for i, score in enumerate(scores):
                mapped_scores.append({
                    "tag": TAG_LABELS[i],
                    "score": float(score)
                })

            mapped_scores.sort(key=lambda x: x["score"], reverse=True)
            top_3 = mapped_scores[:3]

            final_results.append({
                "filename": os.path.basename(path),
                "top_moods": top_3
            })

        except Exception as e:
            print(f"'{path}' 분석 중 에러: {e}")

    return final_results


if __name__ == "__main__":
    test_photos = [f"photo{i}.png" for i in range(1, 42)]

    results = analyze_images_individually(test_photos)

    for res in results:
        print(f"사진: [{res['filename']}]")
        for item in res['top_moods']:
            print(f"  - #{item['tag']} ({item['score'] * 100:.1f}%)")
        print("-" * 30)
