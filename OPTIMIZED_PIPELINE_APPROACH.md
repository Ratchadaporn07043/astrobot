# 🚀 แนวทางปรับปรุง Pipeline: บันทึกทีละหน้า vs บันทึกทีเดียว

## 📊 เปรียบเทียบทั้งสองแบบ

### แบบที่ 1: บันทึกทีเดียว (แบบปัจจุบัน)
```python
# ประมวลผลทุกหน้า → เก็บใน memory
for page_num in range(total_pages):
    page_results = process_single_page(...)
    all_chunks.extend(...)

# บันทึกทีเดียว
collection.insert_many(all_chunks)
```

**ข้อดี:**
- ✅ เร็วที่สุด (insert_many เร็วกว่า insert_one มาก)
- ✅ Network overhead ต่ำ (เชื่อมต่อครั้งเดียว)
- ✅ Transaction safety

**ข้อเสีย:**
- ❌ เสี่ยงข้อมูลหาย (crash = เสียข้อมูลทั้งหมด)
- ❌ ใช้ memory สูง
- ❌ ไม่เห็น progress

---

### แบบที่ 2: บันทึกทีละหน้า (แนะนำ)
```python
# เปิด connection ครั้งเดียว
client = MongoClient(MONGO_URL)
db_original = client[ORIGINAL_DB_NAME]
db_summary = client[SUMMARY_DB_NAME]

for page_num in range(total_pages):
    page_results = process_single_page(...)
    
    # บันทึกทันที
    if page_results['text_chunks']:
        db_original[ORIGINAL_TEXT_COLLECTION].insert_many(page_results['text_chunks'])
    if page_results['text_processed_chunks']:
        db_summary[PROCESSED_TEXT_COLLECTION].insert_many(page_results['text_processed_chunks'])
    # ... บันทึกทุกประเภท
    
    print(f"✅ บันทึกหน้า {page_num + 1} แล้ว")

client.close()
```

**ข้อดี:**
- ✅ ข้อมูลปลอดภัย (crash = ข้อมูลที่บันทึกแล้วยังอยู่)
- ✅ ใช้ memory ต่ำ
- ✅ เห็น progress
- ✅ Resume ได้

**ข้อเสีย:**
- ⚠️ ช้ากว่าเล็กน้อย (แต่ยังใช้ insert_many ต่อหน้า)

---

## 🎯 แนวทางที่แนะนำ: **Hybrid Approach (บันทึกแบบ Batch ทุก N หน้า)**

### แนวคิด:
- ประมวลผลและบันทึกทุก 5-10 หน้า (batch)
- ใช้ connection เดียวตลอด
- เก็บ progress เพื่อ resume ได้

### ตัวอย่างโค้ด:

```python
def main_optimized():
    """Pipeline ที่ปรับปรุงแล้ว: บันทึกแบบ batch ทุก N หน้า"""
    
    # === INITIALIZATION ===
    pymupdf_doc = fitz.open(PDF_PATH)
    pdfplumber_pdf = pdfplumber.open(PDF_PATH)
    ocr_reader = get_ocr_reader()
    
    total_pages = len(pymupdf_doc)
    BATCH_SIZE = 5  # บันทึกทุก 5 หน้า
    
    # เปิด MongoDB connection ครั้งเดียว
    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    
    db_original = client[ORIGINAL_DB_NAME]
    db_summary = client[SUMMARY_DB_NAME]
    
    # Collections
    orig_text_col = db_original[ORIGINAL_TEXT_COLLECTION]
    orig_image_col = db_original[ORIGINAL_IMAGE_COLLECTION]
    orig_table_col = db_original[ORIGINAL_TABLE_COLLECTION]
    
    proc_text_col = db_summary[PROCESSED_TEXT_COLLECTION]
    proc_image_col = db_summary[PROCESSED_IMAGE_COLLECTION]
    proc_table_col = db_summary[PROCESSED_TABLE_COLLECTION]
    
    # ลบข้อมูลเก่า (ครั้งเดียวตอนเริ่ม)
    orig_text_col.delete_many({})
    orig_image_col.delete_many({})
    orig_table_col.delete_many({})
    proc_text_col.delete_many({})
    proc_image_col.delete_many({})
    proc_table_col.delete_many({})
    
    # Batch buffers
    batch_text_chunks = []
    batch_image_chunks = []
    batch_table_chunks = []
    batch_text_processed = []
    batch_image_processed = []
    batch_table_processed = []
    
    doc_id_counter = 1
    
    try:
        # === PROCESS PAGES ===
        for page_num in range(total_pages):
            print(f"\n📄 ประมวลผลหน้า {page_num + 1}/{total_pages}")
            
            # ประมวลผลหน้าเดียว
            page_results = process_single_page(
                page_num=page_num,
                pymupdf_page=pymupdf_doc[page_num],
                pdfplumber_pdf=pdfplumber_pdf,
                ocr_reader=ocr_reader,
                doc_id_counter=doc_id_counter
            )
            
            # เพิ่มเข้า batch buffers
            batch_text_chunks.extend(page_results['text_chunks'])
            batch_image_chunks.extend(page_results['image_chunks'])
            batch_table_chunks.extend(page_results['table_chunks'])
            batch_text_processed.extend(page_results['text_processed_chunks'])
            batch_image_processed.extend(page_results['image_processed_chunks'])
            batch_table_processed.extend(page_results['table_processed_chunks'])
            
            # บันทึกเมื่อครบ batch หรือเป็นหน้าสุดท้าย
            if (page_num + 1) % BATCH_SIZE == 0 or page_num == total_pages - 1:
                print(f"💾 บันทึก batch (หน้า {page_num + 1 - len(batch_text_chunks) + 1} - {page_num + 1})...")
                
                # เพิ่ม created_at ให้ทุก chunk
                now = datetime.now()
                for chunk in batch_text_chunks + batch_image_chunks + batch_table_chunks:
                    chunk['created_at'] = now
                for chunk in batch_text_processed + batch_image_processed + batch_table_processed:
                    chunk['created_at'] = now
                
                # บันทึก Original Data
                if batch_text_chunks:
                    orig_text_col.insert_many(batch_text_chunks)
                    print(f"   ✅ บันทึก {len(batch_text_chunks)} text chunks (original)")
                
                if batch_image_chunks:
                    orig_image_col.insert_many(batch_image_chunks)
                    print(f"   ✅ บันทึก {len(batch_image_chunks)} image chunks (original)")
                
                if batch_table_chunks:
                    orig_table_col.insert_many(batch_table_chunks)
                    print(f"   ✅ บันทึก {len(batch_table_chunks)} table chunks (original)")
                
                # บันทึก Processed Data
                if batch_text_processed:
                    proc_text_col.insert_many(batch_text_processed)
                    print(f"   ✅ บันทึก {len(batch_text_processed)} text chunks (processed)")
                
                if batch_image_processed:
                    proc_image_col.insert_many(batch_image_processed)
                    print(f"   ✅ บันทึก {len(batch_image_processed)} image chunks (processed)")
                
                if batch_table_processed:
                    proc_table_col.insert_many(batch_table_processed)
                    print(f"   ✅ บันทึก {len(batch_table_processed)} table chunks (processed)")
                
                # ล้าง batch buffers
                batch_text_chunks.clear()
                batch_image_chunks.clear()
                batch_table_chunks.clear()
                batch_text_processed.clear()
                batch_image_processed.clear()
                batch_table_processed.clear()
                
                # ตรวจสอบ memory
                check_memory()
        
        print("\n✅ Pipeline เสร็จสิ้น!")
        
    except Exception as e:
        print(f"❗ Error: {e}")
        print(f"⚠️ ข้อมูลที่บันทึกไปแล้วยังอยู่ (หน้า 1 - {page_num})")
        raise
    finally:
        # ปิด connection
        client.close()
        pymupdf_doc.close()
        pdfplumber_pdf.close()
```

---

## 📈 เปรียบเทียบประสิทธิภาพ

| เกณฑ์ | แบบที่ 1 (ทีเดียว) | แบบที่ 2 (ทีละหน้า) | Hybrid (Batch) |
|------|-------------------|---------------------|----------------|
| **ความเร็ว** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **ความปลอดภัย** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Memory Usage** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Progress Visibility** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Resume Capability** | ❌ | ✅ | ✅ |

---

## 🎯 สรุป: แนะนำใช้ **Hybrid Approach**

### เหตุผล:
1. ✅ **ประสิทธิภาพดี**: ใช้ `insert_many()` แบบ batch (เร็วกว่า insert_one)
2. ✅ **ข้อมูลปลอดภัย**: บันทึกทุก 5 หน้า (crash = เสียข้อมูลแค่ 0-4 หน้าสุดท้าย)
3. ✅ **ใช้ memory ต่ำ**: ไม่ต้องเก็บทุกหน้าไว้ใน memory
4. ✅ **เห็น progress**: รู้ว่าบันทึกไปแล้วกี่หน้า
5. ✅ **Resume ได้**: สามารถเช็กว่าบันทึกไปถึงหน้าไหนแล้ว

### การปรับ BATCH_SIZE:
- **BATCH_SIZE = 1**: บันทึกทุกหน้า (ปลอดภัยที่สุด แต่ช้าที่สุด)
- **BATCH_SIZE = 5-10**: สมดุลระหว่างความเร็วและความปลอดภัย (แนะนำ)
- **BATCH_SIZE = 100**: เร็วที่สุด แต่เสี่ยงข้อมูลหายมาก

---

## 🔧 Implementation Tips

1. **Connection Pooling**: ใช้ connection เดียวตลอด (ไม่เปิด/ปิดทุก batch)
2. **Error Handling**: จับ exception และบันทึก progress
3. **Progress Tracking**: เก็บ progress ในไฟล์หรือ database เพื่อ resume
4. **Memory Management**: ล้าง batch buffers หลังบันทึก

