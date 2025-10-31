import os
import io
import base64
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import torch
import easyocr
from openai import OpenAI
from datetime import datetime
import json
import gc
import psutil
import re

# 🆕 เพิ่ม PyThaiNLP สำหรับปรับปรุง OCR
try:
    from pythainlp import word_tokenize
    from pythainlp.spell import correct
    from pythainlp.util import normalize
    PYTHAINLP_AVAILABLE = True
    print("✅ PyThaiNLP loaded successfully")
except ImportError:
    PYTHAINLP_AVAILABLE = False
    print("⚠️ PyThaiNLP not available, using basic text processing")

# ✅ แก้ไขปัญหา MPS device, PIL.ANTIALIAS และ tokenizers parallelism
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

# ✅ โหลด .env
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path)

# ✅ ตัวแปรระบบ
PDF_PATH = "data/attention.pdf"
MONGO_URL = os.getenv("MONGO_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUMMARY_DB_NAME = "astrobot_summary"  # สำหรับเก็บข้อมูลที่ summary และ summary embedding แล้ว
ORIGINAL_DB_NAME = "astrobot_original"  # สำหรับเก็บไฟล์ต้นฉบับที่ extract แล้ว

# ✅ ตัวแปรระบบ - Collection Names
# สำหรับข้อมูลต้นฉบับ (ORIGINAL_DB_NAME)
ORIGINAL_TEXT_COLLECTION = "original_text_chunks"
ORIGINAL_IMAGE_COLLECTION = "original_image_chunks"
ORIGINAL_TABLE_COLLECTION = "original_table_chunks"

# สำหรับข้อมูลที่ประมวลผลแล้ว (SUMMARY_DB_NAME)
PROCESSED_TEXT_COLLECTION = "processed_text_chunks"
PROCESSED_IMAGE_COLLECTION = "processed_image_chunks"
PROCESSED_TABLE_COLLECTION = "processed_table_chunks"

# ✅ ฟังก์ชันตรวจสอบ memory
def check_memory():
    """ตรวจสอบการใช้ memory"""
    memory = psutil.virtual_memory()
    print(f"💾 Memory: {memory.percent}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
    if memory.percent > 80:
        print("⚠️ High memory usage, running garbage collection...")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# 🆕 ฟังก์ชันปรับปรุงข้อความไทยจาก OCR ด้วย PyThaiNLP
def improve_thai_ocr_text(ocr_text):
    """
    ปรับปรุงข้อความไทยจาก OCR ด้วย PyThaiNLP
    """
    if not PYTHAINLP_AVAILABLE or not ocr_text.strip():
        return ocr_text
    
    try:
        # ทำความสะอาดข้อความ
        text = ocr_text.strip()
        
        # แก้ไขการเว้นวรรคที่ผิด
        text = re.sub(r'([ก-๙])([A-Za-z])', r'\1 \2', text)  # เว้นวรรคระหว่างไทย-อังกฤษ
        text = re.sub(r'([A-Za-z])([ก-๙])', r'\1 \2', text)  # เว้นวรรคระหว่างอังกฤษ-ไทย
        text = re.sub(r'([ก-๙])([0-9])', r'\1 \2', text)    # เว้นวรรคระหว่างไทย-ตัวเลข
        text = re.sub(r'([0-9])([ก-๙])', r'\1 \2', text)    # เว้นวรรคระหว่างตัวเลข-ไทย
        
        # แก้ไขการเว้นวรรคที่ซ้ำ
        text = re.sub(r'\s+', ' ', text)
        
        # แบ่งคำด้วย PyThaiNLP
        words = word_tokenize(text, engine='newmm')
        
        # แก้ไขคำผิดด้วย PyThaiNLP
        corrected_words = []
        for word in words:
            if len(word) > 2 and word.isalpha():  # แก้ไขเฉพาะคำที่มีความยาวมากกว่า 2 ตัวอักษร
                try:
                    corrected = correct(word)
                    corrected_words.append(corrected if corrected else word)
                except:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        # รวมคำกลับเป็นประโยค
        improved_text = ' '.join(corrected_words)
        
        # ทำความสะอาดอีกครั้ง
        improved_text = re.sub(r'\s+', ' ', improved_text).strip()
        
        return improved_text
        
    except Exception as e:
        print(f"⚠️ Error in Thai text improvement: {e}")
        return ocr_text

# ✅ โหลดโมเดลแบบ lazy loading
def get_embedding_model():
    """โหลด embedding model แบบ lazy loading"""
    if not hasattr(get_embedding_model, 'model'):
        print("🔄 Loading embedding model...")
        get_embedding_model.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return get_embedding_model.model

def get_semantic_model():
    """โหลด semantic model แบบ lazy loading"""
    if not hasattr(get_semantic_model, 'model'):
        print("🔄 Loading semantic model...")
        get_semantic_model.model = SentenceTransformer("minishlab/potion-multilingual-128M", device="cpu")
    return get_semantic_model.model

def get_ocr_reader():
    """โหลด OCR reader แบบ lazy loading"""
    if not hasattr(get_ocr_reader, 'reader'):
        print(" Loading OCR reader...")
        get_ocr_reader.reader = easyocr.Reader(['en', 'th'], gpu=False, verbose=False)
    return get_ocr_reader.reader

# OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ อ่านข้อความจาก PDF ด้วย PyMuPDF
def extract_text_with_pymupdf(path):
    """
    อ่านข้อความจาก PDF ด้วย PyMuPDF
    """
    print(f"📖 กำลังอ่านข้อความจาก: {path}")
    text_output = ""
    doc = fitz.open(path)
    
    try:
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            if page_text.strip():
                text_output += f"\n--- หน้า {page_num + 1} ---\n{page_text}"
            
            # ตรวจสอบ memory ทุก 20 หน้า
            if page_num % 20 == 0:
                check_memory()
                
    finally:
        doc.close()
    
    return text_output

# ✅ แปลงรูปภาพเป็นข้อความด้วย OCR + PyThaiNLP (ปรับปรุง memory management)
def extract_images_with_ocr(path):
    """
    แปลงรูปภาพใน PDF เป็นข้อความด้วย OCR + PyThaiNLP
    """
    print(f"กำลังแปลงรูปภาพเป็นข้อความจาก: {path}")
    images_data = []
    doc = fitz.open(path)
    
    try:
        ocr_reader = get_ocr_reader()
        
        for page_num, page in enumerate(doc):
            images = page.get_images(full=True)
            print(f"หน้า {page_num + 1}: {len(images)} รูป")
            
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # ตรวจสอบขนาดรูปภาพ
                    image = Image.open(io.BytesIO(image_bytes))
                    width, height = image.size
                    
                    # ข้ามรูปที่ใหญ่เกินไป
                    if width * height > 1500000:  # 1.5M pixels
                        print(f"⚠️ ข้ามรูปใหญ่ {img_index + 1} ({width}x{height})")
                        continue
                    
                    # ข้ามรูปที่เล็กเกินไป
                    if width < 50 or height < 50:
                        print(f"⚠️ ข้ามรูปเล็ก {img_index + 1} ({width}x{height})")
                        continue
                    
                    # OCR
                    ocr_results = ocr_reader.readtext(image_bytes)
                    ocr_text = " ".join([result[1] for result in ocr_results if result[2] > 0.3])  # ลด confidence threshold
                    
                    if ocr_text.strip():
                        # 🆕 ปรับปรุงข้อความด้วย PyThaiNLP
                        improved_text = improve_thai_ocr_text(ocr_text)
                        
                        image_info = {
                            "page": page_num + 1,
                            "image_index": img_index + 1,
                            "original_text": ocr_text.strip(),
                            "improved_text": improved_text,
                            "text": improved_text,  # ใช้ข้อความที่ปรับปรุงแล้ว
                            "image_base64": base64.b64encode(image_bytes).decode("utf-8")
                        }
                        images_data.append(image_info)
                        
                        print(f"✅ รูป {img_index + 1}: {len(improved_text)} ตัวอักษร")
                    
                    # ล้าง memory
                    del image, image_bytes, ocr_results
                    
                except Exception as e:
                    print(f"❗ Error processing image {img_index + 1} on page {page_num + 1}: {e}")
                    continue
            
            # ตรวจสอบ memory หลังจากประมวลผลแต่ละหน้า
            if page_num % 5 == 0:
                check_memory()
            
            # จำกัดจำนวนรูปต่อหน้า
            if len(images_data) > 50:  # จำกัดไม่เกิน 50 รูป
                print("⚠️ จำกัดจำนวนรูปที่ 50 รูป")
                break
                
    finally:
        doc.close()
    
    return images_data

# ✅ แปลงตารางเป็นข้อความด้วย pdfplumber
def extract_tables_with_pdfplumber(path):
    """
    แปลงตารางใน PDF เป็นข้อความด้วย pdfplumber
    """
    print(f" กำลังแปลงตารางเป็นข้อความจาก: {path}")
    tables_data = []
    
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table_index, table in enumerate(tables):
                    if table:
                        # แปลงตารางเป็นข้อความ
                        table_text = ""
                        for row in table:
                            if row:
                                row_text = " | ".join([cell if cell else "" for cell in row])
                                table_text += row_text + "\n"
                        
                        if table_text.strip():
                            table_info = {
                                "page": page_num + 1,
                                "table_index": table_index + 1,
                                "text": table_text.strip()
                            }
                            tables_data.append(table_info)
                
                # ตรวจสอบ memory ทุก 10 หน้า
                if page_num % 10 == 0:
                    check_memory()
                    
    except Exception as e:
        print(f"❗ Error extracting tables: {e}")
    
    return tables_data

# ✅ Semantic Chunking ด้วย Potion Model
def semantic_chunking_with_potion(text, content_type, chunk_size=1000, overlap=200):
    """
    แบ่งข้อความด้วย Semantic Chunking โดยใช้ Potion Model
    """
    print(f"🧠 เริ่ม Semantic Chunking สำหรับ {content_type.upper()}")
    
    try:
        semantic_model = get_semantic_model()
        
        # แบ่งข้อความเป็นประโยค
        sentences = text.split('. ')
        if len(sentences) <= 1:
            return [{"text": text, "type": content_type, "chunk_id": 0}]
        
        # จำกัดจำนวนประโยคเพื่อประหยัด memory
        if len(sentences) > 500:
            sentences = sentences[:500]
            print(f"⚠️ จำกัดที่ 500 ประโยคเพื่อประหยัด memory")
        
        # สร้าง embeddings สำหรับประโยค
        sentence_embeddings = semantic_model.encode(sentences)
        
        # คำนวณความคล้ายคลึงระหว่างประโยค
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            current_length += len(sentence)
            
            # ตรวจสอบว่าควรแบ่ง chunk หรือไม่
            if current_length >= chunk_size or i == len(sentences) - 1:
                chunk_text = '. '.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "type": content_type,
                    "chunk_id": len(chunks)
                })
                current_chunk = []
                current_length = 0
        
        # ล้าง memory
        del sentence_embeddings, sentences
        check_memory()
        
        return chunks
        
    except Exception as e:
        print(f"❗ Error in semantic chunking: {e}")
        # Fallback: แบ่งแบบธรรมดา
        return [{"text": text, "type": content_type, "chunk_id": 0}]

# ✅ สร้าง Embeddings
def create_embeddings(text):
    """
    สร้าง embeddings สำหรับข้อความ
    """
    try:
        embedding_model = get_embedding_model()
        return embedding_model.encode(text).tolist()
    except Exception as e:
        print(f"❗ Error creating embeddings: {e}")
        return [0.0] * 384  # fallback vector

# ✅ สรุปข้อความด้วย OpenAI
def summarize_with_openai(text, content_type):
    """
    สรุปข้อความด้วย OpenAI GPT
    """
    try:
        prompt = f"""
        สรุปเนื้อหาต่อไปนี้ให้กระชับและเข้าใจง่าย (ภาษาไทย):
        
        ประเภทเนื้อหา: {content_type}
        เนื้อหา: {text[:2000]}...
        
        กรุณาสรุปให้ไม่เกิน 3 ประโยค
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=150,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❗ Error in summarization: {e}")
        return text[:200] + "..." if len(text) > 200 else text

# ✅ บันทึกข้อมูลต้นฉบับลง MongoDB (ไม่มี embeddings และ summary)
def store_original_data_in_mongodb(chunks, collection_name):
    """
    บันทึกข้อมูลต้นฉบับลง ORIGINAL_DB_NAME (ไม่มี embeddings และ summary)
    """
    try:
        # ลองเชื่อมต่อ MongoDB Atlas
        print(f"🔗 กำลังเชื่อมต่อ MongoDB Atlas...")
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        
        # ทดสอบการเชื่อมต่อ
        client.admin.command('ping')
        print(f"✅ เชื่อมต่อ MongoDB Atlas สำเร็จ")
        
        # ใช้ ORIGINAL_DB_NAME สำหรับข้อมูลต้นฉบับ
        db_name = ORIGINAL_DB_NAME
        print(f"📊 ใช้ Database: {db_name} (Original - ไม่มี embeddings/summary)")
        
        db = client[db_name]
        collection = db[collection_name]
        
        # ลบข้อมูลเก่า
        collection.delete_many({})
        
        # บันทึกข้อมูลต้นฉบับ (ไม่มี embeddings และ summary)
        for i, chunk in enumerate(chunks):
            print(f"📝 กำลังบันทึกข้อมูลต้นฉบับ chunk {i+1}/{len(chunks)}...")
            
            # สร้างสำเนาของ chunk และเพิ่ม created_at
            original_chunk = chunk.copy()
            original_chunk["created_at"] = datetime.now()
            
            # ไม่เพิ่ม embeddings และ summary
            collection.insert_one(original_chunk)
            
            # ตรวจสอบ memory ทุก 5 chunks
            if i % 5 == 0:
                check_memory()
        
        print(f"✅ บันทึกข้อมูลต้นฉบับ {len(chunks)} chunks ลง {collection_name}")
        client.close()
        
    except Exception as e:
        print(f"❗ MongoDB Atlas connection failed: {e}")
        print(f"💾 บันทึกลงไฟล์ JSON แทน...")
        
        # Fallback: บันทึกลงไฟล์ JSON
        store_original_to_json(chunks, collection_name)

# ✅ บันทึกข้อมูลที่ประมวลผลแล้วลง MongoDB (มี embeddings และ summary)
def store_processed_data_in_mongodb(chunks, collection_name):
    """
    บันทึกข้อมูลที่ประมวลผลแล้วลง SUMMARY_DB_NAME (มี embeddings และ summary)
    """
    try:
        # ลองเชื่อมต่อ MongoDB Atlas
        print(f"🔗 กำลังเชื่อมต่อ MongoDB Atlas...")
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        
        # ทดสอบการเชื่อมต่อ
        client.admin.command('ping')
        print(f"✅ เชื่อมต่อ MongoDB Atlas สำเร็จ")
        
        # ใช้ SUMMARY_DB_NAME สำหรับข้อมูลที่ประมวลผลแล้ว
        db_name = SUMMARY_DB_NAME
        print(f"📊 ใช้ Database: {db_name} (Processed - มี summary embeddings/summary)")
        
        db = client[db_name]
        collection = db[collection_name]
        
        # ลบข้อมูลเก่า
        collection.delete_many({})
        
        # บันทึกข้อมูลที่ประมวลผลแล้ว (มี summary embeddings และ summary)
        for i, chunk in enumerate(chunks):
            print(f"📝 กำลังประมวลผล chunk {i+1}/{len(chunks)}...")
            
            # สร้างสำเนาของ chunk และเพิ่มข้อมูลที่ประมวลผลแล้ว
            processed_chunk = chunk.copy()
            processed_chunk["created_at"] = datetime.now()
            
            # สร้าง summary ก่อน
            summary_text = summarize_with_openai(chunk["text"], chunk["type"])
            processed_chunk["summary"] = summary_text
            
            # สร้าง embeddings จาก summary แทน text ต้นฉบับ
            processed_chunk["embeddings"] = create_embeddings(summary_text)
            
            collection.insert_one(processed_chunk)
            
            # ตรวจสอบ memory ทุก 3 chunks
            if i % 3 == 0:
                check_memory()
        
        print(f"✅ บันทึกข้อมูลที่ประมวลผลแล้ว {len(chunks)} chunks ลง {collection_name}")
        client.close()
        
    except Exception as e:
        print(f"❗ MongoDB Atlas connection failed: {e}")
        print(f"💾 บันทึกลงไฟล์ JSON แทน...")
        
        # Fallback: บันทึกลงไฟล์ JSON
        store_processed_to_json(chunks, collection_name)

# ✅ บันทึกข้อมูลต้นฉบับลงไฟล์ JSON (fallback)
def store_original_to_json(chunks, collection_name):
    """
    บันทึกข้อมูลต้นฉบับลงไฟล์ JSON เป็น fallback (ไม่มี embeddings และ summary)
    """
    try:
        # สร้างโฟลเดอร์ output ถ้าไม่มี
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # บันทึกข้อมูลต้นฉบับ (ไม่มี embeddings และ summary)
        original_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"📝 กำลังบันทึกข้อมูลต้นฉบับ chunk {i+1}/{len(chunks)}...")
            
            # สร้างสำเนาของ chunk และเพิ่ม created_at
            original_chunk = chunk.copy()
            original_chunk["created_at"] = datetime.now().isoformat()
            
            # ไม่เพิ่ม embeddings และ summary
            original_chunks.append(original_chunk)
            
            # ตรวจสอบ memory ทุก 5 chunks
            if i % 5 == 0:
                check_memory()
        
        # บันทึกลงไฟล์
        filename = f"{output_dir}/{collection_name}_original.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(original_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"✅ บันทึกข้อมูลต้นฉบับ {len(original_chunks)} chunks ลง {filename}")
        
    except Exception as e:
        print(f"❗ Error saving original data to JSON: {e}")

# ✅ บันทึกข้อมูลที่ประมวลผลแล้วลงไฟล์ JSON (fallback)
def store_processed_to_json(chunks, collection_name):
    """
    บันทึกข้อมูลที่ประมวลผลแล้วลงไฟล์ JSON เป็น fallback (มี embeddings และ summary)
    """
    try:
        # สร้างโฟลเดอร์ output ถ้าไม่มี
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # ประมวลผล chunks
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"📝 กำลังประมวลผล chunk {i+1}/{len(chunks)}...")
            
            # สร้างสำเนาของ chunk และเพิ่มข้อมูลที่ประมวลผลแล้ว
            processed_chunk = chunk.copy()
            processed_chunk["created_at"] = datetime.now().isoformat()
            
            # สร้าง summary ก่อน
            summary_text = summarize_with_openai(chunk["text"], chunk["type"])
            processed_chunk["summary"] = summary_text
            
            # สร้าง embeddings จาก summary แทน text ต้นฉบับ
            processed_chunk["embeddings"] = create_embeddings(summary_text)
            processed_chunks.append(processed_chunk)
            
            # ตรวจสอบ memory ทุก 3 chunks
            if i % 3 == 0:
                check_memory()
        
        # บันทึกลงไฟล์
        filename = f"{output_dir}/{collection_name}_processed.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(processed_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"✅ บันทึกข้อมูลที่ประมวลผลแล้ว {len(processed_chunks)} chunks ลง {filename}")
        
    except Exception as e:
        print(f"❗ Error saving processed data to JSON: {e}")

# ✅ ฟังก์ชันประมวลผลหน้าเดียว (ตาม flow ที่ออกแบบ)
def process_single_page(page_num, pymupdf_page, pdfplumber_pdf, ocr_reader, doc_id_counter):
    """
    ประมวลผลหน้าเดียว: Extract → Summary → Embedding → Store
    ตาม flow ที่ออกแบบ: ทำหน้าแล้วทำ summary/embedding ทันที
    
    Args:
        page_num: หมายเลขหน้าที่กำลังประมวลผล (0-based)
        pymupdf_page: หน้า PDF จาก PyMuPDF
        pdfplumber_pdf: PDF object จาก pdfplumber
        ocr_reader: OCR reader สำหรับประมวลผลรูปภาพ
        doc_id_counter: counter สำหรับสร้าง doc_id
        
    Returns:
        dict: {
            'has_content': bool,  # มีเนื้อหาหรือไม่ (สำหรับตรวจสอบหน้าเปล่า)
            'text_chunks': list,
            'image_chunks': list,
            'table_chunks': list,
            'text_processed_chunks': list,
            'image_processed_chunks': list,
            'table_processed_chunks': list
        }
    """
    page_results = {
        'has_content': False,
        'text_chunks': [],
        'image_chunks': [],
        'table_chunks': [],
        'text_processed_chunks': [],
        'image_processed_chunks': [],
        'table_processed_chunks': []
    }
    
    try:
        print(f"\n{'='*50}")
        print(f"📄 กำลังประมวลผลหน้า {page_num + 1}")
        print(f"{'='*50}")
        
        # === EXTRACT: Text, Images, Tables จากหน้าเดียวกัน ===
        # 1. Extract Text (เจออะไรก่อนทำอันนั้น)
        page_text = pymupdf_page.get_text("text")
        if page_text.strip():
            page_results['has_content'] = True
            print(f"✅ พบ Text: {len(page_text)} ตัวอักษร")
            
            # Create text chunk
            text_chunk = {
                "text": page_text.strip(),
                "type": "text",
                "chunk_id": len(page_results['text_chunks']),
                "page": page_num + 1,
                "doc_id": f"doc_{doc_id_counter}_{page_num + 1}_text"
            }
            page_results['text_chunks'].append(text_chunk)
            
            # สร้าง summary และ embeddings ทันที
            summary_text = summarize_with_openai(text_chunk["text"], "text")
            text_processed_chunk = text_chunk.copy()
            text_processed_chunk["summary"] = summary_text
            text_processed_chunk["embeddings"] = create_embeddings(summary_text)
            text_processed_chunk["created_at"] = datetime.now()
            page_results['text_processed_chunks'].append(text_processed_chunk)
            print(f"   ✅ สร้าง summary และ embeddings แล้ว")
        
        # 2. Extract Images (เจออะไรก่อนทำอันนั้น)
        images = pymupdf_page.get_images(full=True)
        print(f"🔍 พบ {len(images)} รูปในหน้านี้")
        
        for img_index, img in enumerate(images):
            try:
                xref = img[0]
                base_image = pymupdf_page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                
                # ตรวจสอบขนาดรูปภาพ
                image = Image.open(io.BytesIO(image_bytes))
                width, height = image.size
                
                # ข้ามรูปที่ใหญ่เกินไป
                if width * height > 1500000:
                    print(f"⚠️ ข้ามรูปใหญ่ {img_index + 1} ({width}x{height})")
                    continue
                
                # ข้ามรูปที่เล็กเกินไป
                if width < 50 or height < 50:
                    print(f"⚠️ ข้ามรูปเล็ก {img_index + 1} ({width}x{height})")
                    continue
                
                # OCR
                ocr_results = ocr_reader.readtext(image_bytes)
                ocr_text = " ".join([result[1] for result in ocr_results if result[2] > 0.3])
                
                if ocr_text.strip():
                    page_results['has_content'] = True
                    # ปรับปรุงข้อความด้วย PyThaiNLP
                    improved_text = improve_thai_ocr_text(ocr_text)
                    
                    print(f"✅ รูป {img_index + 1}: {len(improved_text)} ตัวอักษร")
                    
                    # Create image chunk
                    image_chunk = {
                        "text": improved_text,
                        "type": "image",
                        "chunk_id": len(page_results['image_chunks']),
                        "page": page_num + 1,
                        "image_index": img_index + 1,
                        "original_text": ocr_text.strip(),
                        "improved_text": improved_text,
                        "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
                        "doc_id": f"doc_{doc_id_counter}_{page_num + 1}_img_{img_index + 1}"
                    }
                    page_results['image_chunks'].append(image_chunk)
                    
                    # สร้าง summary และ embeddings ทันที
                    summary_text = summarize_with_openai(image_chunk["text"], "image")
                    image_processed_chunk = image_chunk.copy()
                    image_processed_chunk["summary"] = summary_text
                    image_processed_chunk["embeddings"] = create_embeddings(summary_text)
                    image_processed_chunk["created_at"] = datetime.now()
                    page_results['image_processed_chunks'].append(image_processed_chunk)
                    print(f"   ✅ สร้าง summary และ embeddings แล้ว")
                
                # ล้าง memory
                del image, image_bytes, ocr_results
                
            except Exception as e:
                print(f"❗ Error processing image {img_index + 1} on page {page_num + 1}: {e}")
                continue
        
        # 3. Extract Tables (เจออะไรก่อนทำอันนั้น)
        if page_num < len(pdfplumber_pdf.pages):
            pdfplumber_page = pdfplumber_pdf.pages[page_num]
            tables = pdfplumber_page.extract_tables()
            print(f"🔍 พบ {len(tables)} ตารางในหน้านี้")
            
            for table_index, table in enumerate(tables):
                if table:
                    # แปลงตารางเป็นข้อความ
                    table_text = ""
                    for row in table:
                        if row:
                            row_text = " | ".join([cell if cell else "" for cell in row])
                            table_text += row_text + "\n"
                    
                    if table_text.strip():
                        page_results['has_content'] = True
                        print(f"✅ ตาราง {table_index + 1}: {len(table_text)} ตัวอักษร")
                        
                        # Create table chunk
                        table_chunk = {
                            "text": table_text.strip(),
                            "type": "table",
                            "chunk_id": len(page_results['table_chunks']),
                            "page": page_num + 1,
                            "table_index": table_index + 1,
                            "doc_id": f"doc_{doc_id_counter}_{page_num + 1}_table_{table_index + 1}"
                        }
                        page_results['table_chunks'].append(table_chunk)
                        
                        # สร้าง summary และ embeddings ทันที
                        summary_text = summarize_with_openai(table_chunk["text"], "table")
                        table_processed_chunk = table_chunk.copy()
                        table_processed_chunk["summary"] = summary_text
                        table_processed_chunk["embeddings"] = create_embeddings(summary_text)
                        table_processed_chunk["created_at"] = datetime.now()
                        page_results['table_processed_chunks'].append(table_processed_chunk)
                        print(f"   ✅ สร้าง summary และ embeddings แล้ว")
        
        # สรุปผลการประมวลผลหน้า
        if not page_results['has_content']:
            print(f"⚠️ หน้า {page_num + 1} เป็นหน้าเปล่า (ไม่มี text, images, หรือ tables)")
        else:
            total_chunks = (len(page_results['text_chunks']) + 
                          len(page_results['image_chunks']) + 
                          len(page_results['table_chunks']))
            print(f"✅ ประมวลผลหน้า {page_num + 1} เสร็จ: {total_chunks} chunks")
        
        return page_results
        
    except Exception as e:
        print(f"❗ Error processing page {page_num + 1}: {e}")
        return page_results

# ✅ ฟังก์ชันหลัก (แก้ไขให้ตรงกับ flow ที่ออกแบบ)
def main():
    print("🚀 เริ่ม Pipeline: Extract → OCR + PyThaiNLP → Summary → Embedding → Store")
    print("📄 ประมวลผลหน้าแล้วทำ Summary/Embedding ทันที (ตาม Flow)")
    print()
    
    try:
        # === INITIALIZATION ===
        print("=== INITIALIZATION ===")
        check_memory()
        
        # เปิดไฟล์ PDF ทั้ง PyMuPDF และ pdfplumber
        pymupdf_doc = fitz.open(PDF_PATH)
        pdfplumber_pdf = pdfplumber.open(PDF_PATH)
        ocr_reader = get_ocr_reader()
        
        total_pages = len(pymupdf_doc)
        print(f"📚 จำนวนหน้าทั้งหมด: {total_pages} หน้า")
        
        # ตัวแปรสำหรับเก็บผลลัพธ์ทั้งหมด
        all_text_chunks = []
        all_image_chunks = []
        all_table_chunks = []
        all_text_processed_chunks = []
        all_image_processed_chunks = []
        all_table_processed_chunks = []
        
        doc_id_counter = 1  # สำหรับสร้าง doc_id
        
        # === LOOP: More Pages (ตาม flow ที่ออกแบบ) ===
        print("\n=== STEP 1: PAGE-BY-PAGE PROCESSING ===")
        for page_num in range(total_pages):
            # ประมวลผลหน้าเดียว (Extract → Summary → Embedding)
            page_results = process_single_page(
                page_num=page_num,
                pymupdf_page=pymupdf_doc[page_num],
                pdfplumber_pdf=pdfplumber_pdf,
                ocr_reader=ocr_reader,
                doc_id_counter=doc_id_counter
            )
            
            # รวม chunks จากหน้านี้เข้ากับทั้งหมด
            all_text_chunks.extend(page_results['text_chunks'])
            all_image_chunks.extend(page_results['image_chunks'])
            all_table_chunks.extend(page_results['table_chunks'])
            all_text_processed_chunks.extend(page_results['text_processed_chunks'])
            all_image_processed_chunks.extend(page_results['image_processed_chunks'])
            all_table_processed_chunks.extend(page_results['table_processed_chunks'])
            
            # ตรวจสอบ memory ทุก 5 หน้า
            if (page_num + 1) % 5 == 0:
                check_memory()
            
            # ตรวจสอบว่ามีหน้าอื่นอีกไหม (More Pages Decision)
            if page_num < total_pages - 1:
                print(f"➡️ มีหน้าอื่นอีก {total_pages - page_num - 1} หน้า")
            else:
                print(f"✅ ประมวลผลครบทุกหน้าแล้ว ({total_pages} หน้า)")
                print("➡️ ส่งเข้าสู่ SentenceTransformer และ MongoDB collection summary")
        
        # ปิดไฟล์ PDF
        pymupdf_doc.close()
        pdfplumber_pdf.close()
        
        # === STEP 2: STORE IN MONGODB ===
        print("\n=== STEP 2: STORE IN MONGODB ===")
        check_memory()
        
        print(f"\n📊 สรุปผลการประมวลผล:")
        print(f"   📝 Text chunks: {len(all_text_chunks)}")
        print(f"   🖼️ Image chunks: {len(all_image_chunks)}")
        print(f"   📊 Table chunks: {len(all_table_chunks)}")
        print(f"   📊 Processed chunks: {len(all_text_processed_chunks) + len(all_image_processed_chunks) + len(all_table_processed_chunks)}")
        
        # เก็บข้อมูลต้นฉบับใน ORIGINAL_DB_NAME (ไม่มี embeddings และ summary)
        print("\n📁 เก็บข้อมูลต้นฉบับใน ORIGINAL_DB_NAME...")
        if all_text_chunks:
            store_original_data_in_mongodb(all_text_chunks, ORIGINAL_TEXT_COLLECTION)
        if all_image_chunks:
            store_original_data_in_mongodb(all_image_chunks, ORIGINAL_IMAGE_COLLECTION)
        if all_table_chunks:
            store_original_data_in_mongodb(all_table_chunks, ORIGINAL_TABLE_COLLECTION)
        
        # เก็บข้อมูลที่ประมวลผลแล้ว (มี summary embedding และ summary) ใน SUMMARY_DB_NAME
        print("\n📊 เก็บข้อมูลที่ประมวลผลแล้วใน SUMMARY_DB_NAME...")
        print("   (Summary embeddings ถูกสร้างจาก summary text โดยใช้ SentenceTransformer)")
        if all_text_processed_chunks:
            # ใช้ store_processed_data_in_mongodb แต่จะข้ามการสร้าง summary/embeddings 
            # เพราะสร้างไว้แล้วใน process_single_page
            try:
                client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
                client.admin.command('ping')
                db = client[SUMMARY_DB_NAME]
                collection = db[PROCESSED_TEXT_COLLECTION]
                collection.delete_many({})
                collection.insert_many(all_text_processed_chunks)
                print(f"✅ บันทึก {len(all_text_processed_chunks)} processed text chunks")
                client.close()
            except Exception as e:
                print(f"❗ Error storing processed text chunks: {e}")
        
        if all_image_processed_chunks:
            try:
                client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
                client.admin.command('ping')
                db = client[SUMMARY_DB_NAME]
                collection = db[PROCESSED_IMAGE_COLLECTION]
                collection.delete_many({})
                collection.insert_many(all_image_processed_chunks)
                print(f"✅ บันทึก {len(all_image_processed_chunks)} processed image chunks")
                client.close()
            except Exception as e:
                print(f"❗ Error storing processed image chunks: {e}")
        
        if all_table_processed_chunks:
            try:
                client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
                client.admin.command('ping')
                db = client[SUMMARY_DB_NAME]
                collection = db[PROCESSED_TABLE_COLLECTION]
                collection.delete_many({})
                collection.insert_many(all_table_processed_chunks)
                print(f"✅ บันทึก {len(all_table_processed_chunks)} processed table chunks")
                client.close()
            except Exception as e:
                print(f"❗ Error storing processed table chunks: {e}")
        
        print("\n✅ Pipeline เสร็จสิ้น!")
        print(f"✅ ข้อมูลทั้งหมดถูกบันทึกใน MongoDB:")
        print(f"   - Original: {ORIGINAL_DB_NAME}")
        print(f"   - Summary: {SUMMARY_DB_NAME}")
        
    except Exception as e:
        print(f"❗ Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()
        print("🔄 Running garbage collection...")
        gc.collect()
        check_memory()

if __name__ == "__main__":
    main()