"""
Projenin Amacı:
Bu modül; yüzlerce sayfalık karmaşık teknik katalogları (yapı kimyasalları vb.) okuyup 
vektör veritabanına dönüştüren ve son kullanıcıların "Hangi zemine ne uygulamalıyım?" 
gibi spesifik sorularına RAG (Retrieval-Augmented Generation) mimarisiyle 
saniyeler içinde net cevaplar üreten ana yapay zeka motorudur.
Kurumsal veri güvenliği gereği marka isimlerinden bağımsız tasarlanmıştır.
"""

from google import genai
from google.genai import types
import chromadb
from pypdf import PdfReader
import os
import time

# Konfigürasyon ve Veritabanı Bağlantıları
import os

# Security: API anahtarını çevre değişkeninden almak güvenlik best-practice'idir.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("HATA: GEMINI_API_KEY bulunamadı! Lütfen ortam değişkenlerini kontrol edin.")

client = genai.Client(api_key=GEMINI_API_KEY)
chroma_client = chromadb.PersistentClient(path="veritabani")
koleksiyon = chroma_client.get_or_create_collection(name="belgelerim")

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"

# Belge İşleme ve Parçalama Motoru

def pdf_oku(dosya_yolu):
    """PDF belgesini ayrıştırarak metin çıkarır."""
    reader = PdfReader(dosya_yolu)
    metin = ""
    for sayfa in reader.pages:          # Sayfa nesneleri üzerinde yineleme yapar
        metin += sayfa.extract_text()   # Metin verisini ayıklar ve birleştirir
    return metin

def metni_parcala(metin, max_harf=500, kesisme_payi=50):
    """
    Uzun metinleri analiz için overlap (örtüşme) stratejisiyle chunk'lara ayırır.
    Context bütünlüğünü (örn: kelime kopmalarını) korumak için kesisme_payi kullanılır.
    """
    parcalar = []
    baslangic = 0
    
    while baslangic < len(metin):
        bitis = baslangic + max_harf
        parca = metin[baslangic:bitis]
        parcalar.append(parca)
        baslangic += (max_harf - kesisme_payi)
    
    return parcalar

# Vektörleştirme Mimarisi (Embeddings)

def embedding_al(metin, task_type="retrieval_document"):
    """API üzerinden metin için embedding vektörü oluşturur (Rate limit yönetimi dahil)."""
    while True:
        try:
            sonuc = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=metin,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            return sonuc.embeddings[0].values
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("  ⏳ Rate limit doldu, 15 saniye bekleniyor...")
                time.sleep(15)  # Bekle ve tekrar dene
            else:
                raise  # Farklı bir hata ise üstle ilet

def belge_yukle(dosya_yolu):
    """Belge işleme hattı: Okuma, chunking, vektörleştirme ve indeksleme."""
    metin = pdf_oku(dosya_yolu)
    parcalar = metni_parcala(metin)
    
    # Caching (Önbellekleme): Belge zaten indekslenmişse işlemi atla
    if koleksiyon.get(ids=[f"{dosya_yolu}_0"])["ids"]:
        print(f"✅ '{dosya_yolu}' zaten yüklü — atlandı")
        return
    
    # Text parçalarını (chunks) veritabanına kaydet
    print(f"📂 '{dosya_yolu}' işleniyor ve indeksleniyor ({len(parcalar)} parça)...")
    yeni = 0
    for i, parca in enumerate(parcalar):
        if len(parca.strip()) < 50:
            continue
        vektor = embedding_al(parca)
        koleksiyon.add(documents=[parca], embeddings=[vektor], ids=[f"{dosya_yolu}_{i}"])
        yeni += 1
        print(f"  Parça {i+1}/{len(parcalar)} kaydedildi")
        time.sleep(0.7)
    print(f"✅ Yüklendi: {yeni} parça")

# RAG (Retrieval-Augmented Generation) Sorgulama Mimarisi

def soru_sor(soru):
    """Veritabanı üzerinde semantik arama yapar ve LLM aracılığıyla yanıt üretir."""
    
    soru_vektoru = embedding_al(soru, task_type="retrieval_query")
    
    sonuclar = koleksiyon.query(
        query_embeddings=[soru_vektoru],
        n_results=3
    )
    
    ilgili_parcalar = sonuclar["documents"][0]
    context = "\n---\n".join(ilgili_parcalar)
    
    prompt = f"""Sen bir teknik asistansın.
Sadece aşağıdaki belge parçalarına dayanarak cevap ver.
Belgede olmayan bir bilgiyi kesinlikle uydurma.
Emin değilsen "Bu bilgi belgelerimde yer almıyor" de.

BELGELER:
{context}

SORU: {soru}"""
    
    # LLM İstek Yönetimi (Exponential Backoff ile)
    while True:
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt
            )
            return response.text
        except Exception as e:
            hata = str(e)
            if "503" in hata or "UNAVAILABLE" in hata:
                print("  ⏳ Sunucu meşgul, 10 saniye bekleniyor...")
                time.sleep(10)
            elif "429" in hata or "RESOURCE_EXHAUSTED" in hata:
                print("  ⏳ Kota doldu, 30 saniye bekleniyor...")
                time.sleep(30)
            else:
                raise

# Ana Dosya Çalıştırma Bloğu

if __name__ == "__main__":
    
    # Yerel belge dizinindeki dokümanları tara ve işle
    belgeler_klasoru = "belgeler"
    pdf_sayisi = 0
    
    for dosya in os.listdir(belgeler_klasoru):
        if dosya.endswith(".pdf"):
            belge_yukle(os.path.join(belgeler_klasoru, dosya))
            pdf_sayisi += 1
    
    if pdf_sayisi == 0:
        print("⚠️  'belgeler' klasöründe PDF bulunamadı!")
    
    toplam = koleksiyon.count()
    print(f"\n📚 Veritabanında toplam {toplam} parça mevcut.")
    print("Sistem hazır. Çıkmak için 'q' yaz.\n")
    
    while True:
        soru = input("Soru: ")
        
        if soru.lower() == "q":
            break
            
        cevap = soru_sor(soru)
        print(f"\nCevap: {cevap}\n")