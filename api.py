from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
import chromadb
import os
import time

# Konfigürasyon: API anahtarını güvenli şekilde ortam değişkeninden al veya sunucuda run edilecekse veriable kısmına ekle

# GEMINI_API_KEY = "" 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("HATA: GEMINI_API_KEY çevre değişkeni bulunamadı! Lütfen Railway'de veya .env dosyasında ayarlayın.")

client = genai.Client(api_key=GEMINI_API_KEY)
chroma = chromadb.PersistentClient(path="veritabani")
koleksiyon = chroma.get_or_create_collection("belgelerim")

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL       = "gemini-2.5-flash"

# FastAPI Uygulama ve Middleware Tanımlamaları
app = FastAPI(title="RAG Asistan API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Geliştirme için açık — prodüksiyonda URL'yi yaz
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temel RAG (Retrieval-Augmented Generation) Servisleri

def embedding_al(metin: str, task_type: str = "retrieval_document"):
    """Metni vektöre çevirir (Exponential backoff uygular)."""
    max_deneme = 3
    for deneme in range(max_deneme):
        try:
            sonuc = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=metin,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            return sonuc.embeddings[0].values
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "503" in str(e):
                if deneme < max_deneme - 1:
                    time.sleep(2 ** (deneme + 1))
                    continue
            raise e

def soru_sor(soru: str):
    """RAG pipeline: embed → retrieve → generate stream"""
    # 1. Metni vektörel uzaya çevir ve en yakın k-dökümanı (top-k) sorgula
    soru_vektoru = embedding_al(soru, task_type="retrieval_query")

    sonuclar = koleksiyon.query(
        query_embeddings=[soru_vektoru],
        n_results=3
    )
    ilgili_parcalar = sonuclar["documents"][0]
    context = "\n---\n".join(ilgili_parcalar)

    prompt = f"""Sen KYK Yapı Kimyasalları'nın profesyonel ve teknik akıllı asistanısın.
Görevlerin:
1. Sana verilen bağlam (context) bilgisini kullanarak müşterinin sorusunu Türkçe yanıtla.
2. Kesinlikle "Belgelere göre", "Belgelerinizde yazıyor", "Bana verdiğiniz bilgiye göre" gibi ifadeler KULLANMA. Doğrudan şirket çalışanı/uzmanı gibi doğrudan bilgi ver.
3. Bağlamda cevaba dair hiçbir şey yoksa, "Ürünlerimiz arasında buna uygun spesifik bir çözüm bulamadım, detaylar için müşteri hizmetleriyle iletişime geçebilirsiniz." şeklinde profesyonelce yanıtla, asla uydurma.
4. Müşteri özel bir şehirden (örn. Mardin) bahsederse, oranın iklim özelliklerine (sıcak vb.) göre mevcut ürünleri mantıklıca yorumla ama "Şu şehir belgede geçmiyor" gibi tuhaf cümleler kurma.

BAĞLAM BİLGİSİ (Sadece bu ürünleri/bilgileri kullan):
{context}

MÜŞTERİ SORUSU: {soru}"""

    # 2. LLM akışı (Streaming & Exponential Backoff)
    max_deneme = 3
    for deneme in range(max_deneme):
        try:
            response = client.models.generate_content_stream(
                model=LLM_MODEL,
                contents=prompt
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            return
        except Exception as e:
            hata = str(e)
            if "503" in hata or "UNAVAILABLE" in hata or "429" in hata or "RESOURCE_EXHAUSTED" in hata:
                if deneme < max_deneme - 1:
                    time.sleep(2 ** (deneme + 1))
                    continue
                else:
                    yield "⚠️ HATA: Google yapay zeka sunucuları şu an aşırı yoğun veya yanıt vermiyor. Lütfen daha sonra tekrar deneyin."
                    return
            else:
                yield f"⚠️ HATA: Bir sistem hatası oluştu: {hata}"
                return

# API Uç Noktaları (Endpoints)
class SoruIstek(BaseModel):
    soru: str

@app.post("/sor")
async def sor_endpoint(istek: SoruIstek):
    """Soru endpoint'i (StreamingResponse döner)"""
    if not istek.soru.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz")
    
    return StreamingResponse(
        soru_sor(istek.soru),
        media_type="text/plain"
    )

@app.get("/istatistik")
async def istatistik():
    """Veritabanı istatistiklerini döner."""
    return {
        "toplam_parca": koleksiyon.count(),
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }

@app.get("/saglik")
async def saglik():
    """Health check — Railway bu endpoint'i kontrol eder."""
    return {"durum": "çalışıyor"}

# Statik dosyaların (Frontend) uygulamaya dahil edilmesi (Mount işlemi)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
