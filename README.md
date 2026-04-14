# Kurumsal Teknik Servis - RAG Akıllı Asistan API

Bu proje, mimarların, bayilerin ve mühendislerin yüzlerce sayfalık teknik dokümanlar arasında boğulmasını önlemek amacıyla geliştirilmiş **Yapay Zeka (AI) destekli bir RAG (Retrieval-Augmented Generation)** arama ve sentez motorudur.

## 🚀 Projenin Amacı ve Vizyonu

Güncel endüstriyel ürün katalogları ve teknik uygulama rehberleri çok spesifik çevresel şartlar (sıcaklık, zemin türü, nem oranı) barındırır. Bu sistemin vizyonu, kullanıcının "Sıcak bir iklimde dış cephe için ne kullanmalıyım?" sorusuna karşılık, arka planda kapsamlı kurumsal teknik ürün rehberini tarayıp saniyeler içinde anında, anlaşılır ve %100 dokümana sadık teknik bir cevap üretmesidir.

[![Canlı Demo](https://img.shields.io/badge/Canl%C4%B1%20Demo-%C4%B0ncelemek%20%C4%B0%C3%A7in%20T%C4%B1kla-success?style=for-the-badge&logo=vercel&logoColor=white)](https://kyk-demo-production.up.railway.app/)

## 🛠️ Teknoloji Stack'i

Bu sistem, veri madenciliği ve modern backend endüstrisinin güncel araçları kullanılarak **Ahmet Akaslan** tarafından tasarlanmış ve geliştirilmiştir:

- **FastAPI (Python):** Asenkron çağrılar ve streaming (akış) özellikleri ile yüksek performanslı API mimarisi.
- **ChromaDB:** PDF içeriklerinden çıkarılan metinlerin (chunk) anlamsal uzayda aratılabilmesi için optimize edilmiş yerel Vektör Veritabanı.
- **Google Gemini 2.5 Flash:** Veritabanından (Chroma) "cımbızlanan" teknik paragrafları insan doğallığında sentezleyen LLM motoru.
- **Vanilla RAG Mimarisini:** Yapay zekaya veri "ezberletmek" (Fine-Tuning) yerine "açık kitap sınavı" yaptırarak halüsinasyon riskini sıfıra indiren kurumsal yapı.
- **Streaming response (ReadableStream):** Frontend'de ChatGPT tarzı anlık kelime animasyonlu (UX) veri gösterimi.

## ⚙️ Kurulum ve Canlıya Alma (Deployment)

Projenin internete açık bir Web servisi olarak çalıştırılması için hantal VPS metotları yerine modern geliştiricilerin tercihi olan **Railway.app** kullanılmıştır. Yayınlamak son derece basittir:

1. Bu projeyi kendi GitHub hesabınıza aktarın (fork / push).
2. [Railway.app](https://railway.app/) üzerinden GitHub reposunu projeye bağlayın.
3. Güvenlik ve gizlilik best-practice'leri gereği, Railway'in `Variables` bölümünden `GEMINI_API_KEY` atamasını gerçekleştirin.
4. Railway projeyi otomatik olarak izole bir konteynerde çalıştıracak ve global erişim linkini saniyeler içinde sizinle paylaşacaktır.

> **Local (Yerel) Geliştirme İçin:** Bilgisayarınızda test etmek isterseniz bir `.env` dosyası oluşturup Google Cloud üzerinden aldığınız API anahtarını içerisine eklemeniz yeterlidir.

---
*Geliştirici: **Ahmet Akaslan***
*2026*
