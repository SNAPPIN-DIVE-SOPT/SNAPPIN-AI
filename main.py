import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os

model = SentenceTransformer('clip-ViT-B-32')

IMAGE_TAG_DEFINITIONS = {
  "내추럴": "candid snapshot, person not posing for camera, looking away or unaware of camera, unposed moment, walking or mid-motion, off-center framing, asymmetrical composition, natural daylight, documentary photography",
  "연출된": "posed portrait, direct eye contact with camera, intentional pose, fixed posture, centered subject, symmetrical composition, clearly staged scene, deliberate setup",
  "서사적": "person interacting with objects, action in progress, contextual background, storytelling scene, cinematic still, narrative moment",
  "따스한": "warm sunlight, golden hour, sunset light, yellow-orange tone, soft shadows, natural warm light",
  "청량한": "bright daylight, clear blue sky, vivid green nature, outdoor scene, open space, fresh air feeling, high visibility",
  "투명한": "very bright exposure, white or pale background, extremely low saturation, flat lighting, almost no shadow, washed-out colors, airy and clean look",
  "몽환적인": "dreamy atmosphere, strong backlight, light flare, glow effect, soft focus, low contrast, ethereal mood",
  "뚜렷한": "sharp focus, strong contrast, clear subject separation, defined edges, crisp details, strong visual center",
  "차가운": "cool tone, blue light, night scene, artificial lighting, neon or fluorescent light, strong shadows, urban night mood",
  "디지털": "digital photography, ultra sharp focus, high clarity, precise edges, clean texture, no grain, realistic color reproduction, modern camera look",
  "아날로그": "film photography, visible film grain, soft focus, low clarity, muted or faded colors, uneven exposure, nostalgic film look, imperfect texture",
  "Y2K": "direct flash photography, point-and-shoot camera style, harsh flash shadows, overexposed highlights, strong color contrast, early 2000s snapshot aesthetic, kitschy vibe"
}

TAG_LABELS = list(IMAGE_TAG_DEFINITIONS.keys())
TAG_TEXTS = list(IMAGE_TAG_DEFINITIONS.values())

def analyze_images_individually(image_paths):
    tag_embeddings = model.encode(TAG_TEXTS)
    final_results = []

    print(f"총 {len(image_paths)}장의 사진 개별 분석 시작 (영어 상세 설명 기반)...\n")

    for path in image_paths:
        try:
            if not os.path.exists(path):
                print(f"파일 없음: {path}")
                continue

            img = Image.open(path)
            img_emb = model.encode(img)

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
