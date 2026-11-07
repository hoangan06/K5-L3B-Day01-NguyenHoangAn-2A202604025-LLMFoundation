# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604025
- **Họ và tên:** Nguyễn Hoàng Ân
- **SEED:** 4620 (thử thách #1)

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra (15/15 check đạt, đủ 10 dự đoán) | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước (5/5 đáp án khớp) | ☑ |

## 2. Kết quả chính

| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.857 | 0.919 | 0.38 | 0 |
| 2. BoW + MLP | 0.915 | 0.974 | 2.83 | 0 |
| 3a. MiniLM + kNN | 0.913 | 0.974 | 74.22 | 0 |
| 3b. MiniLM + LogReg | 0.929 | 0.974 | 42.59 | 0 |
| 3c. MiniLM + MLP head | **0.954** | 0.974 | 38.34 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-6-luna) | – | 0.000* | 1149.16 | 0.0104 |
| 4b. LLM lớn zero-shot (gpt-6.1-terra) | – | 0.000* | 1012.60 | 0.1043 |
| 4c. LLM nhỏ few-shot | – | 0.000* | 838.37 | 0.0222 |

\* Cả ba lần chạy LLM đều `invalid = 40/40`: `parse_label` không đọc được nhãn nào nên mọi tin bị gán `ham`. Prompt đã được viết lại (định nghĩa rõ spam/ham, yêu cầu "một từ viết thường, không giải thích") nhưng vẫn không hết invalid. Đây là lỗi pipeline, không phản ánh năng lực phân loại của LLM; nghi vấn chính là giới hạn `max_completion_tokens=3`, cần in `response_text` thô để xác nhận.

## 3. Dự đoán và thực tế

1. **`hard_eval`**: Tôi đoán BoW khó nhất với *Tưởng spam* và *Nói về spam*. Thực tế BoW đúng 90–100% ở hai nhóm này và sai nặng nhất ở *Spam giọng tự nhiên* (0%), *Thao túng/lừa đảo* (12–25%) và *Né từ khoá* (17%). Tôi lệch vì nghĩ từ khoá spam trong tin ham sẽ gây báo nhầm, nhưng các từ ngôi thứ nhất như `i`, `my`, `me` mang trọng số ham đủ mạnh để kéo lại. Điểm yếu thật của BoW là chiều ngược lại: thiếu từ khoá thì bỏ lọt.
2. **`llm_models`**: Tôi đoán model lớn có F1 cao hơn và LLM thắng 3b, nhưng cả hai đều F1 = 0 vì 40/40 invalid. Riêng phần chi phí thì đúng: 0.1043 / 0.0104 = **10 lần**, bằng đúng tỉ lệ giá mỗi token.
3. **`tradeoff`**: Tôi đoán Pareto gồm BoW+LogReg và Encoder+MLP, và Encoder+MLP vượt SLA. Thực tế Pareto chỉ có BoW+LogReg và BoW+MLP, còn 3c đạt SLA (p95 38 ms). Trên SUB 40, BoW+MLP có cùng F1 0.974 với 3a/3b/3c nhưng nhanh hơn khoảng 14–26 lần ở p95, nên lấn át cả ba. Kết luận này phụ thuộc vào tập SUB quá nhỏ: một tin sai làm F1 đổi khoảng 2.5 điểm.

## 4. Phân tích 50 case khó

BoW tụt mạnh nhất: **BoW + LogReg** từ F1 0.919 (SUB) xuống 0.286, **BoW + MLP** từ 0.974 xuống 0.368, accuracy chỉ khoảng 50%. Embedding + kNN bền nhất (acc 68%, Thao túng 100%, Né từ khoá 83%). Không ghép nào thắng được *Spam giọng tự nhiên*: có 6 case mà mọi model đều sai. Hàng LLM (acc 42%, tức đoán toàn `ham`) không dùng để kết luận vì lỗi parse ở mục 2.

- **#14** *"Hi, I'm working from home and making about £300 a day, happy to show you how…"* (mọi model đều sai): tin không có từ khoá spam nào trong top trọng số dương (`call`, `free`, `txt`…), trong khi `i` và các từ giọng kể chuyện đều mang trọng số kéo về ham, nên logit âm. Embedding cũng đặt câu này gần vùng "tin cá nhân", vì nó mã hoá *giọng văn* chứ không mã hoá *ý đồ* lừa việc làm.
- **Nhóm Né từ khoá** (BoW chỉ đúng 1/6), ví dụ *"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w…"*: tokenizer biến `c0ngr4tulat10ns`, `w0n`, `cl4im` thành các token ngoài vocab, nên vector gần như bằng 0. MiniLM tách subword nên vẫn nhận ra dạng "congratulations / won / claim"; nhờ vậy nhóm này 3a đạt 83%, 3b/3c đạt 67%.

## 5. Thử thách cá nhân

**Đề #1:** đổi ngưỡng quyết định của mô hình Phần 1 từ 0.5 sang 0.2 và 0.8. Cách làm: tính `predict_proba(X_te, w, b)` một lần, rồi áp từng ngưỡng và gọi `metrics`.

| Ngưỡng | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.2 | 0.899 | 0.899 | **0.899** |
| 0.5 | 0.987 | 0.758 | 0.857 |
| 0.8 | **1.000** | 0.566 | 0.723 |

Hạ ngưỡng thì gán spam dễ hơn, nên recall tăng và precision giảm; nâng ngưỡng thì ngược lại. Ngưỡng 0.2 cho F1 tốt nhất, nhưng với ngân hàng sợ chặn nhầm OTP (false positive) tôi chọn **0.8**. Trên tập test có 99 spam, ngưỡng 0.8 bỏ lọt 43 tin (ở 0.5 là 24 tin) để đổi lấy 0 tin bị chặn nhầm (ở 0.5 là 1 tin). Mất một tin OTP đắt hơn nhiều so với lọt một tin spam. Lưu ý precision = 1.0 chỉ đo trên 775 tin test, chưa đảm bảo không chặn nhầm khi chạy thật.

## 6. Phương án cho sản phẩm thật

Bốn cách đạt SLA p95 < 50 ms là BoW+LogReg, BoW+MLP, 3b và 3c. Tôi chọn **3c (MiniLM + MLP head)** vì nó có F1 test cao nhất trong nhóm này (0.954, p95 38.3 ms) và bền hơn BoW trên case khó (acc 58% so với 50–52%). Đánh đổi là throughput: 3c chỉ xử lý khoảng 32 tin/s mỗi tiến trình, trong khi 10 triệu SMS/ngày cần trung bình 116 tin/s, nên phải chạy ít nhất 4 worker (nhiều hơn vào giờ cao điểm) hoặc dùng GPU.

Để giảm tải, tôi đề xuất **router hai tầng**. Tầng 1 là BoW + LogReg cho mọi tin (p95 0.38 ms, gần như miễn phí): p < 0.2 gán ham, p > 0.8 gán spam (ở ngưỡng 0.8 precision trên test là 1.0). Chỉ những tin lưỡng lự mới chuyển lên tầng 2 là 3c.

LLM bị loại: p95 từ 0.84 đến 1.15 s, vượt SLA khoảng 17–23 lần. Chi phí cũng cao: theo tính tay, 1 triệu tin tốn $10.5/ngày, nên 10 triệu tin tốn khoảng $105/ngày, tức **~$3.150/tháng**. Con số đo được ($0.0104/1k tin, tức ~$104/ngày) khớp với ước tính này.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời/viết gì | Tôi sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 4 (4.1–4.4) | _(điền prompt đã dùng)_ | _(điền)_ | Viết lại prompt 4.2 (thêm định nghĩa spam/ham, "một từ viết thường"); chạy bộ test giả lập ✅ 4.1 + 4.4, 4.2–4.3 |
| Báo cáo | Tổng hợp notebook + `lab1_report.json` thành báo cáo theo mẫu | Bản nháp các mục 2–6 | Đối chiếu từng số với output notebook |

**Điều học được:** Điểm cao trên tập test cùng phân phối không chứng minh model hiểu bài toán: BoW đạt F1 0.92 trên SUB nhưng chỉ còn 0.29 khi gặp spam viết tự nhiên, và nó yếu ở chỗ thiếu từ khoá chứ không phải ở chỗ thừa từ khoá như tôi tưởng. Ngưỡng quyết định là một nút điều chỉnh nghiệp vụ, không phải hằng số 0.5. Cuối cùng, pipeline có thể hỏng âm thầm (40/40 invalid vẫn cho ra một con số F1), nên phải xem output thô trước khi tin vào metric.
