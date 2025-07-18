### pipeline của 2 mô hình LLM

Bert Model: [pipeline_bert](https://drive.google.com/drive/folders/19gVt4VBPBq2HasQxvw0wRPsYJ-IDUqMR?usp=drive_link)

XLNet Model: [pipeline_xlent](https://drive.google.com/drive/folders/13wgIZQpbukjTM9BkyU5N55DnbY6Vlm13?usp=drive_link)
# Fake News Detection Dataset

##  Giới thiệu

Bộ dữ liệu được xây dựng phục vụ cho bài toán **phát hiện tin giả (Fake News Detection)** hoặc **nhận diện thông tin sai lệch (Misinformation Detection)**.

Dữ liệu được chia thành 2 tập:
- `DataSet_Misinfo_FAKE`: Bao gồm các văn bản/tin tức được xác định là **giả mạo (FAKE)**. [data_fake](https://drive.google.com/file/d/1RiZvNZgw9oJOSyjlhWX0j8O1W4dJIzgE/view?usp=drive_link)
- `DataSet_Misinfo_TRUE`: Bao gồm các văn bản/tin tức được xác nhận là **sự thật (TRUE)**. [data_true](https://drive.google.com/file/d/1RiZvNZgw9oJOSyjlhWX0j8O1W4dJIzgE/view?usp=drive_link)

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
  - TF-IDF/CountVectorizer + Logistic Regression / SVM / Random Forest/Decision Tree/Native Bayes.
- Mô hình học sâu:
  - BERT, XLNet.

##  Phân chia dữ liệu

Tổng số mẫu tùy thuộc vào từng tập:
- FAKE: 43642 samples
- TRUE: 34975 samples
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
  - Decision Tree
  - Random Forest.
  - Naive Bayes.
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
## Kết quả thu được.
<img width="1519" height="589" alt="image" src="https://github.com/user-attachments/assets/c268cac9-27a4-415b-833d-f9808dfe7225" />

<img width="1519" height="295" alt="image" src="https://github.com/user-attachments/assets/ced898b3-aeaa-4f7b-824f-ff8ba06dbb69" />
## Kết luận
Trong quá trình đánh giá các mô hình học máy truyền thống và mô hình học sâu trên tập dữ liệu phát hiện tin giả, kết quả cho thấy sự khác biệt rõ rệt về hiệu quả giữa hai nhóm mô hình.

Ở nhóm mô hình truyền thống, Logistic Regression đạt kết quả cao nhất với độ chính xác 95.50%, cho thấy khả năng phân loại khá ổn định và phù hợp với bài toán. Mô hình SVM (SVC) cũng cho kết quả tương đối tốt với độ chính xác 94.15%. Trong khi đó, Random Forest đạt mức 92.79%, thể hiện tính ổn định hơn so với Decision Tree đơn lẻ. Ngược lại, Multinomial Naive Bayes và Decision Tree có độ chính xác thấp nhất, lần lượt là 88.40% và 89.67%, cho thấy chúng không thực sự phù hợp với đặc thù dữ liệu văn bản trong bài toán này.

Với nhóm mô hình học sâu, cả hai mô hình BERT-BASE-CASED và XLNET-BASE-CASED đều đạt độ chính xác rất cao, lên tới 99.33%. Điều này khẳng định ưu thế vượt trội của các mô hình Transformer trong việc xử lý và phân tích dữ liệu văn bản, nhờ khả năng hiểu ngữ cảnh sâu sắc và trích xuất đặc trưng ngôn ngữ hiệu quả.

Nhìn chung, mô hình BERT và XLNet là hai phương án tối ưu và phù hợp nhất cho bài toán phát hiện tin giả, trong khi Logistic Regression có thể được lựa chọn khi cần một mô hình truyền thống đơn giản mà vẫn đảm bảo độ chính xác tương đối cao.
