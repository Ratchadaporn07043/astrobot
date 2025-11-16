#!/usr/bin/env python3
"""
ค้นหาหน้าใน PDF ที่มีรูปภาพ
"""
import os
import fitz  # PyMuPDF

PDF_PATH = "data/attention.pdf"

if not os.path.exists(PDF_PATH):
    print(f"❌ ไม่พบไฟล์: {PDF_PATH}")
    exit(1)

print(f"🔍 กำลังค้นหาหน้าที่มีรูปภาพใน: {PDF_PATH}")
print()

doc = fitz.open(PDF_PATH)
total_pages = len(doc)
pages_with_images = []

for page_num in range(min(50, total_pages)):  # ตรวจสอบ 50 หน้าแรก
    page = doc[page_num]
    images = page.get_images(full=True)
    if len(images) > 0:
        pages_with_images.append((page_num + 1, len(images)))
        print(f"✅ หน้า {page_num + 1}: พบ {len(images)} รูปภาพ")

doc.close()

if pages_with_images:
    print(f"\n📊 สรุป: พบ {len(pages_with_images)} หน้าที่มีรูปภาพ")
    print(f"   หน้าแรกที่มีรูปภาพ: หน้า {pages_with_images[0][0]} ({pages_with_images[0][1]} รูป)")
else:
    print(f"\n⚠️ ไม่พบรูปภาพใน 50 หน้าแรก")

