# 🖼️ แนวทางปรับปรุงการ Embed รูปภาพแผนภูมิโหราศาสตร์

## 📊 สถานะปัจจุบัน

### ✅ สิ่งที่ทำได้:
1. **OCR + Text Embedding**
   - แปลงข้อความในรูปภาพเป็นข้อความด้วย EasyOCR
   - สร้าง summary ด้วย OpenAI
   - สร้าง text embeddings จาก summary

### ❌ สิ่งที่ทำไม่ได้:
1. **Image Embedding** - ไม่มีการใช้ Vision Model เพื่อดึงคุณสมบัติทางสายตาของรูปภาพ
2. **Structured Data Extraction** - ไม่มีการสกัดข้อมูลเชิงโครงสร้างจากแผนภูมิ (ตำแหน่งดาว, มุมสัมพันธ์)

---

## 🚀 แนวทางแก้ไข

### 1. เพิ่ม Image Embedding ด้วย Vision Model

#### ตัวเลือกที่ 1: ใช้ OpenAI Vision API (แนะนำ)
```python
from openai import OpenAI

def create_image_embedding_with_openai(image_bytes):
    """
    สร้าง image embedding ด้วย OpenAI Vision API
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # แปลง image bytes เป็น base64
    import base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # ใช้ Vision API เพื่อสร้าง embedding
    response = client.embeddings.create(
        model="text-embedding-3-large",  # หรือใช้ vision model
        input=image_base64
    )
    
    return response.data[0].embedding
```

#### ตัวเลือกที่ 2: ใช้ CLIP Model (Open Source)
```python
import torch
from transformers import CLIPProcessor, CLIPModel

def get_clip_model():
    """โหลด CLIP model แบบ lazy loading"""
    if not hasattr(get_clip_model, 'model'):
        print("🔄 Loading CLIP model...")
        get_clip_model.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        get_clip_model.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return get_clip_model.model, get_clip_model.processor

def create_image_embedding_with_clip(image_bytes):
    """
    สร้าง image embedding ด้วย CLIP
    """
    model, processor = get_clip_model()
    
    # แปลง image bytes เป็น PIL Image
    from PIL import Image
    import io
    image = Image.open(io.BytesIO(image_bytes))
    
    # ประมวลผลด้วย CLIP
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    return image_features[0].tolist()
```

#### ตัวเลือกที่ 3: ใช้ Sentence Transformers (Multimodal)
```python
from sentence_transformers import SentenceTransformer

def get_multimodal_model():
    """โหลด multimodal model"""
    if not hasattr(get_multimodal_model, 'model'):
        print("🔄 Loading multimodal model...")
        # ใช้โมเดลที่รองรับทั้ง text และ image
        get_multimodal_model.model = SentenceTransformer('clip-ViT-B-32')
    return get_multimodal_model.model

def create_image_embedding_with_sentence_transformers(image_bytes):
    """
    สร้าง image embedding ด้วย Sentence Transformers
    """
    model = get_multimodal_model()
    
    # แปลง image bytes เป็น PIL Image
    from PIL import Image
    import io
    image = Image.open(io.BytesIO(image_bytes))
    
    # สร้าง embedding
    embedding = model.encode(image)
    return embedding.tolist()
```

---

### 2. เพิ่ม Structured Data Extraction

#### ใช้ GPT-4 Vision เพื่อสกัดข้อมูลเชิงโครงสร้าง
```python
def extract_astrological_structure_with_gpt4v(image_bytes):
    """
    ใช้ GPT-4 Vision เพื่อสกัดข้อมูลเชิงโครงสร้างจากแผนภูมิโหราศาสตร์
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    import base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = """
    กรุณาสกัดข้อมูลเชิงโครงสร้างจากแผนภูมิโหราศาสตร์นี้:
    
    1. ตำแหน่งดาวเคราะห์แต่ละดวง (ราศี, องศา, ลิปดา)
    2. ตำแหน่ง House Cusps (จุดเริ่มต้นของแต่ละเรือนชะตา)
    3. มุมสัมพันธ์ (Aspects) ระหว่างดาวเคราะห์
    
    กรุณาแปลงข้อมูลเป็น JSON format:
    {
        "planets": [
            {"name": "Sun", "sign": "Aquarius", "degree": 18, "minute": 30, "house": 1},
            ...
        ],
        "houses": [
            {"number": 1, "cusp_sign": "Aquarius", "cusp_degree": 18},
            ...
        ],
        "aspects": [
            {"planet1": "Sun", "planet2": "Moon", "aspect": "Conjunction", "orb": 5.2},
            ...
        ]
    }
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",  # หรือ gpt-4-vision-preview
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=2000
    )
    
    # Parse JSON response
    import json
    structure_data = json.loads(response.choices[0].message.content)
    
    # แปลงเป็นข้อความสำหรับ embedding
    structure_text = convert_structure_to_text(structure_data)
    
    return structure_data, structure_text

def convert_structure_to_text(structure_data):
    """
    แปลงข้อมูลเชิงโครงสร้างเป็นข้อความสำหรับ embedding
    """
    text_parts = []
    
    # Planets
    for planet in structure_data.get("planets", []):
        text_parts.append(
            f"{planet['name']} อยู่ในราศี{planet['sign']} "
            f"องศา {planet['degree']}°{planet.get('minute', 0)}' "
            f"เรือนชะตาที่ {planet['house']}"
        )
    
    # Aspects
    for aspect in structure_data.get("aspects", []):
        text_parts.append(
            f"{aspect['planet1']} ทำมุม {aspect['aspect']} "
            f"กับ {aspect['planet2']} (orb: {aspect['orb']}°)"
        )
    
    return ". ".join(text_parts)
```

---

### 3. แก้ไขโค้ดใน `multimodel_rag.py`

#### เพิ่ม Image Embedding ใน process_single_page
```python
# ในส่วนที่ประมวลผล image
elif element_type == 'image':
    # ... OCR code ...
    
    if ocr_text.strip():
        # สร้าง image chunk
        image_chunk = {
            "text": improved_text,
            "type": "image",
            # ... existing fields ...
        }
        
        # เพิ่ม: สร้าง image embedding
        print(f"   🔄 กำลังสร้าง image embedding...")
        image_embedding = create_image_embedding_with_clip(image_bytes)
        image_chunk["image_embedding"] = image_embedding
        
        # เพิ่ม: สกัดข้อมูลเชิงโครงสร้าง (ถ้าเป็นแผนภูมิโหราศาสตร์)
        if is_astrological_chart(image_bytes):  # ตรวจสอบว่าเป็นแผนภูมิหรือไม่
            print(f"   🔄 กำลังสกัดข้อมูลเชิงโครงสร้าง...")
            structure_data, structure_text = extract_astrological_structure_with_gpt4v(image_bytes)
            image_chunk["structure_data"] = structure_data
            image_chunk["structure_text"] = structure_text
            
            # สร้าง embedding จาก structure text
            structure_embedding = create_embeddings(structure_text)
            image_chunk["structure_embedding"] = structure_embedding
        
        # สร้าง summary และ text embedding (เหมือนเดิม)
        summary_text = summarize_with_openai(image_chunk["text"], "image")
        text_embedding = create_embeddings(summary_text)
        
        # เก็บทั้ง text embedding และ image embedding
        image_processed_chunk = image_chunk.copy()
        image_processed_chunk["summary"] = summary_text
        image_processed_chunk["text_embeddings"] = text_embedding
        image_processed_chunk["image_embeddings"] = image_embedding
```

---

## 📋 สรุป

### สำหรับแผนภูมิโหราศาสตร์:

1. **ข้อความที่ OCR อ่านได้** → ✅ Embed ได้ (Text Embedding)
2. **คุณสมบัติทางสายตาของรูปภาพ** → ⚠️ ต้องเพิ่ม Image Embedding
3. **ข้อมูลเชิงโครงสร้าง** → ⚠️ ต้องเพิ่ม Structured Data Extraction

### แนะนำ:
- **ระยะสั้น**: ใช้ OCR + Text Embedding (ทำอยู่แล้ว) → ใช้ได้สำหรับข้อความในแผนภูมิ
- **ระยะกลาง**: เพิ่ม Image Embedding ด้วย CLIP → ดึงคุณสมบัติทางสายตาของแผนภูมิ
- **ระยะยาว**: เพิ่ม Structured Data Extraction ด้วย GPT-4 Vision → ดึงข้อมูลเชิงโครงสร้างที่แท้จริง

---

## 🔧 Implementation Steps

1. ติดตั้ง dependencies:
```bash
pip install transformers torch sentence-transformers
# หรือ
pip install openai  # สำหรับ Vision API
```

2. เพิ่มฟังก์ชันสร้าง image embedding
3. แก้ไข `process_single_page()` เพื่อใช้ image embedding
4. อัปเดต MongoDB schema เพื่อเก็บ image_embeddings
5. อัปเดต retrieval system เพื่อค้นหาทั้ง text และ image embeddings

