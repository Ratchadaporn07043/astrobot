#!/usr/bin/env python3
"""
สคริปต์ทดสอบการสร้าง Image Embedding
"""
import os
import sys
import io
from PIL import Image
from dotenv import load_dotenv

# เพิ่ม path สำหรับ import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# โหลด .env
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print("✅ โหลดไฟล์ .env สำเร็จ")
else:
    print("⚠️ ไม่พบไฟล์ .env")

# Import functions จาก multimodel_rag
from multimodel_rag import (
    get_image_embedding_model,
    create_image_embeddings,
    get_ocr_reader,
    improve_thai_ocr_text
)
import easyocr

def test_image_embedding_from_file(image_path):
    """
    ทดสอบการสร้าง image embedding จากไฟล์รูปภาพ
    """
    print("=" * 60)
    print("🧪 ทดสอบ Image Embedding จากไฟล์")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ ไม่พบไฟล์: {image_path}")
        return False
    
    try:
        # อ่านรูปภาพ
        print(f"\n📖 กำลังอ่านรูปภาพ: {image_path}")
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # แสดงข้อมูลรูปภาพ
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        print(f"📏 ขนาดรูปภาพ: {width}x{height} pixels")
        print(f"📊 Format: {image.format}")
        print(f"💾 ขนาดไฟล์: {len(image_bytes) / 1024:.2f} KB")
        
        # ทดสอบการสร้าง image embedding
        print(f"\n🔄 กำลังสร้าง image embedding...")
        image_embedding = create_image_embeddings(image_bytes)
        
        if image_embedding is not None:
            print(f"✅ สร้าง image embedding สำเร็จ!")
            print(f"   📊 Dimensions: {len(image_embedding)}")
            print(f"   📈 ตัวอย่างค่า (5 ค่าแรก): {image_embedding[:5]}")
            print(f"   📈 ตัวอย่างค่า (5 ค่าสุดท้าย): {image_embedding[-5:]}")
            print(f"   📊 Min value: {min(image_embedding):.6f}")
            print(f"   📊 Max value: {max(image_embedding):.6f}")
            print(f"   📊 Mean value: {sum(image_embedding)/len(image_embedding):.6f}")
            return True
        else:
            print(f"❌ ไม่สามารถสร้าง image embedding ได้")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_embedding_from_pdf(pdf_path, page_num=0):
    """
    ทดสอบการสร้าง image embedding จากรูปภาพใน PDF
    """
    print("\n" + "=" * 60)
    print("🧪 ทดสอบ Image Embedding จาก PDF")
    print("=" * 60)
    
    if not os.path.exists(pdf_path):
        print(f"❌ ไม่พบไฟล์: {pdf_path}")
        return False
    
    try:
        import fitz  # PyMuPDF
        
        # เปิด PDF
        print(f"\n📖 กำลังเปิด PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        
        if page_num >= len(doc):
            print(f"❌ หน้า {page_num} ไม่มีใน PDF (มีทั้งหมด {len(doc)} หน้า)")
            doc.close()
            return False
        
        page = doc[page_num]
        print(f"📄 หน้า {page_num + 1}/{len(doc)}")
        
        # ดึงรูปภาพจากหน้า
        images = page.get_images(full=True)
        print(f"🖼️ พบ {len(images)} รูปภาพในหน้านี้")
        
        if len(images) == 0:
            print("⚠️ ไม่พบรูปภาพในหน้านี้")
            doc.close()
            return False
        
        # ทดสอบรูปภาพแรก
        img_index = 0
        xref = images[img_index][0]
        print(f"\n🖼️ กำลังทดสอบรูปภาพที่ {img_index + 1}...")
        
        # Extract image
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        
        # แสดงข้อมูลรูปภาพ
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        print(f"📏 ขนาดรูปภาพ: {width}x{height} pixels")
        print(f"📊 Format: {base_image.get('ext', 'unknown')}")
        print(f"💾 ขนาดไฟล์: {len(image_bytes) / 1024:.2f} KB")
        
        # ทดสอบ OCR
        print(f"\n🔍 กำลังทดสอบ OCR...")
        ocr_reader = get_ocr_reader()
        ocr_results = ocr_reader.readtext(image_bytes)
        ocr_text = " ".join([result[1] for result in ocr_results if result[2] > 0.3])
        
        if ocr_text.strip():
            print(f"✅ OCR พบข้อความ: {len(ocr_text)} ตัวอักษร")
            print(f"   📝 ตัวอย่างข้อความ: {ocr_text[:100]}...")
            
            # ปรับปรุงข้อความด้วย PyThaiNLP
            improved_text = improve_thai_ocr_text(ocr_text)
            print(f"   📝 ข้อความหลังปรับปรุง: {len(improved_text)} ตัวอักษร")
        else:
            print(f"⚠️ OCR ไม่พบข้อความ")
        
        # ทดสอบการสร้าง image embedding
        print(f"\n🔄 กำลังสร้าง image embedding...")
        image_embedding = create_image_embeddings(image_bytes)
        
        if image_embedding is not None:
            print(f"✅ สร้าง image embedding สำเร็จ!")
            print(f"   📊 Dimensions: {len(image_embedding)}")
            print(f"   📈 ตัวอย่างค่า (5 ค่าแรก): {image_embedding[:5]}")
            print(f"   📈 ตัวอย่างค่า (5 ค่าสุดท้าย): {image_embedding[-5:]}")
            print(f"   📊 Min value: {min(image_embedding):.6f}")
            print(f"   📊 Max value: {max(image_embedding):.6f}")
            print(f"   📊 Mean value: {sum(image_embedding)/len(image_embedding):.6f}")
            
            doc.close()
            return True
        else:
            print(f"❌ ไม่สามารถสร้าง image embedding ได้")
            doc.close()
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """
    ทดสอบการโหลด CLIP model
    """
    print("=" * 60)
    print("🧪 ทดสอบการโหลด CLIP Model")
    print("=" * 60)
    
    try:
        print("\n🔄 กำลังโหลด CLIP model...")
        model = get_image_embedding_model()
        
        if model is not None:
            print("✅ โหลด CLIP model สำเร็จ!")
            print(f"   📊 Model type: {type(model)}")
            return True
        else:
            print("❌ ไม่สามารถโหลด CLIP model ได้")
            print("   💡 อาจจะต้องติดตั้ง sentence-transformers หรือ torch")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 เริ่มทดสอบ Image Embedding")
    print()
    
    # ทดสอบการโหลด model
    model_ok = test_model_loading()
    
    if not model_ok:
        print("\n❌ ไม่สามารถโหลด model ได้ กรุณาตรวจสอบ dependencies")
        print("   💡 ลองติดตั้ง: pip install sentence-transformers torch")
        sys.exit(1)
    
    # ทดสอบจาก PDF (ถ้ามี)
    pdf_path = "data/attention.pdf"
    if os.path.exists(pdf_path):
        print("\n" + "=" * 60)
        response = input(f"ต้องการทดสอบจาก PDF ({pdf_path}) หรือไม่? (y/n): ")
        if response.lower() in ['y', 'yes', 'ใช่']:
            page_num = input("ใส่หมายเลขหน้าที่ต้องการทดสอบ (0 = หน้าแรก, Enter = 0): ")
            try:
                page_num = int(page_num) if page_num.strip() else 0
            except:
                page_num = 0
            test_image_embedding_from_pdf(pdf_path, page_num)
    else:
        print(f"\n⚠️ ไม่พบไฟล์ PDF: {pdf_path}")
    
    # ทดสอบจากไฟล์รูปภาพ (ถ้ามี)
    print("\n" + "=" * 60)
    image_path = input("ใส่ path ของไฟล์รูปภาพที่ต้องการทดสอบ (หรือ Enter เพื่อข้าม): ")
    if image_path.strip() and os.path.exists(image_path):
        test_image_embedding_from_file(image_path)
    elif image_path.strip():
        print(f"❌ ไม่พบไฟล์: {image_path}")
    
    print("\n" + "=" * 60)
    print("✅ ทดสอบเสร็จสิ้น!")
    print("=" * 60)

