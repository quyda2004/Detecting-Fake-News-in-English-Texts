# Fake News Detection Dataset

##  Giới thiệu

Bộ dữ liệu được xây dựng phục vụ cho bài toán **phát hiện tin giả (Fake News Detection)** hoặc **nhận diện thông tin sai lệch (Misinformation Detection)**.

Dữ liệu được chia thành 2 tập:
- `DataSet_Misinfo_FAKE`: Bao gồm các văn bản/tin tức được xác định là **giả mạo (FAKE)**.
- `DataSet_Misinfo_TRUE`: Bao gồm các văn bản/tin tức được xác nhận là **sự thật (TRUE)**.

##  Cấu trúc dữ liệu

Mỗi mẫu dữ liệu bao gồm:
- **text**: Nội dung văn bản (tin tức, đoạn văn).
- **label**: Nhãn phân loại (`FAKE` hoặc `TRUE`).

Ví dụ:

| text                                                   | label |
|--------------------------------------------------------|-------|
| COVID-19 vaccine causes magnetic effects in humans.    | FAKE  |
| WHO confirms the safety of COVID-19 vaccines.          | TRUE  |
| Bill Gates implanted chips via vaccines.               | FAKE  |
| The earth revolves around the sun.                     | TRUE  |

##  Mục tiêu sử dụng

- Phát triển mô hình phân loại tin giả.
- Phát hiện thông tin sai lệch trên mạng xã hội.
- Hỗ trợ hệ thống kiểm chứng tin tức (fact-checking).

##  Ứng dụng mô hình

Một số mô hình có thể sử dụng để huấn luyện:
- Mô hình truyền thống:
  - TF-IDF + Logistic Regression / SVM / Random Forest.
- Mô hình học sâu:
  - BERT, DistilBERT, XLNet, RoBERTa.
  - RNN / CNN + Word2Vec / GloVe embeddings.

##  Phân chia dữ liệu

Tổng số mẫu tùy thuộc vào từng tập:
- FAKE: xxx samples
- TRUE: yyy samples
##  Chiến lược xử lý (2 hướng chính)

###  Hướng 1: Machine Learning truyền thống (TF-IDF / CountVectorizer)

- **Tiền xử lý văn bản**:
  - Chuyển thành chữ thường.
  - Xóa stopwords.
  - Xóa dấu câu, ký tự đặc biệt.
- xóa các kí tự không có trong unicode 8
- **Biểu diễn văn bản**:
  - TF-IDF Vectorizer.
  - CountVectorizer.

- **Thuật toán phân loại**:
  - Logistic Regression.
  - SVM.
  - Random Forest.
  - Naive Bayes.
  - XGBoost.
-sử dụng Gridsearch để tối ưu tham số của cho từng mô hình 
- **Đặc điểm**:
  - Phân tích mức từ hoặc n-gram.
  - Hiệu quả với dữ liệu nhỏ, dễ triển khai.

---

###  Hướng 2: Deep Learning với Pre-trained Models (BERT, XLNet, RoBERTa)

- **Tiền xử lý tối thiểu**:
  - Giữ nguyên văn bản gốc.
  - Tokenize bằng tokenizer của mô hình (BERT, XLNet).
  - Không cần xóa stopword hay tách từ.

- **Sử dụng mô hình học sâu**:
  - BERT (bert-base-uncased).
  - XLNet.
- Sử dụng fine tune 
- **Đặc điểm**:
  - Mô hình hiểu ngữ cảnh tốt hơn.
  - Áp dụng fine-tuning toàn mô hình hoặc chỉ huấn luyện classifier head.
  - Yêu cầu phần cứng mạnh hơn.

