# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604025
- **Họ và tên:** Nguyễn Hoàng Ân
- **SEED:** 4620 (đề thử thách #1)

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra (15/15 check đạt) | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước (5/5 đáp án khớp hash) | ☑ |

## 2. Kết quả chính

| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.857 | 0.919 | 0.08 | 0 |
| 2. BoW + MLP | 0.915 | 0.974 | 0.21 | 0 |
| 3a. MiniLM + kNN | 0.913 | 0.974 | 175.5 | 0 |
| 3b. MiniLM + LogReg | 0.929 | 0.974 | 35.1 | 0 |
| 3c. MiniLM + MLP head | **0.954** | 0.974 | 33.8 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-6-luna) | – | 0.000* | 1985.7 | 0.0055 |
| 4b. LLM lớn zero-shot (gpt-6.1-terra) | – | 0.000* | 1428.4 | 0.0550 |
| 4c. LLM nhỏ few-shot | – | 0.000* | 1912.7 | 0.0172 |

\* `invalid = 40/40`: `parse_label` không đọc được nhãn nào → mọi tin bị gán `ham`. Đây là lỗi pipeline (nhiều khả năng phản hồi bị cắt/rỗng với `max_completion_tokens=3`), **không** phản ánh năng lực LLM; cần in `response_text` thô để xác nhận và chạy lại.

## 3. Dự đoán và thực tế

1. **`llm_models`** — Đoán: model lớn F1 cao hơn, LLM thắng 3b. Thực tế: cả hai F1 = 0, 40/40 invalid. Lệch vì giả định pipeline LLM chạy đúng; bài học là phải kiểm tra output thô trên vài tin trước khi chạy cả loạt.
2. **`tradeoff`** — Đoán: Pareto gồm BoW+LogReg, Encoder+MLP, LLM lớn. Thực tế: chỉ BoW+LogReg và BoW+MLP. Trên SUB 40, 3c có F1 bằng BoW+MLP (0.974) nhưng p95 chậm ~160 lần → bị lấn át; LLM F1 = 0 nên cũng bị lấn át.
3. **`llm_fewshot`** — Đoán: chi phí tăng gấp đôi. Thực tế: 0.0172 / 0.0055 ≈ **3,1 lần**. 4 ví dụ (có 2 tin spam dài) làm prompt dài gấp ~3 lần prompt zero-shot, mà chi phí gần như tỉ lệ với token input.

## 4. Phân tích 50 case khó

BoW tụt mạnh nhất: **BoW + LogReg** F1 0.919 → 0.286, **BoW + MLP** 0.974 → 0.368 (acc ~50%). Hai nhóm BoW gần như bó tay: *Spam giọng tự nhiên* (0%) và *Thao túng/lừa đảo* (12–25%); *Né từ khoá* chỉ 17%. Embedding + kNN bền nhất (acc 68%, né từ khoá 83%). Hàng LLM (42% = đoán toàn `ham`) không dùng để kết luận vì lỗi parse ở mục 2.

- **#14** *"Hi, I'm working from home and making about £300 a day…"* (mọi model đều sai): không có từ khoá spam; các token `i`, `m`, `home`, `happy` mang trọng số âm (kéo về ham) → logit âm. Embedding cũng đặt câu này gần vùng "tin cá nhân" vì giọng kể chuyện.
- **#29** *"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd…"*: tokenizer biến `c0ngr4tulat10ns`, `w0n`, `cl4im` thành token ngoài vocab → vector gần như rỗng, chỉ còn `a`, `at`… → `w·x` ≈ 0, bias âm (do lệch lớp ~13% spam) quyết định → `ham`. Embedding (subword) vẫn nhận ra dạng "congratulations/won/claim" nên làm tốt hơn.

## 5. Thử thách cá nhân

**Đề #1:** đổi ngưỡng quyết định của mô hình Phần 1 từ 0.5 sang 0.2 và 0.8; báo P/R/F1; ngân hàng sợ chặn nhầm OTP nên chọn ngưỡng nào.

| Ngưỡng | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.2 | _(điền)_ | _(điền)_ | _(điền)_ |
| 0.5 | 0.987 | 0.758 | 0.857 |
| 0.8 | _(điền)_ | _(điền)_ | _(điền)_ |

**Kết luận:** chọn ngưỡng **cao (0.8)**. Chặn nhầm OTP là false positive → cần tối đa precision; tăng ngưỡng làm ít tin bị gán spam hơn → precision tăng, recall giảm (chấp nhận lọt thêm spam). _(Ô code thử thách trong notebook còn trống — cần chạy rồi điền số.)_

## 6. Phương án cho sản phẩm thật

Chọn **3c. MiniLM + MLP head**: đạt SLA (p95 33.8 ms < 50 ms) và F1 test cao nhất (0.954), bền hơn BoW trên case khó (acc 58% vs 50–52%), chi phí API = 0. Đánh đổi: throughput chỉ ~39 tin/s/tiến trình trên Colab, trong khi 10 triệu SMS/ngày ≈ 116 tin/s trung bình → cần ≥ 3–4 worker (nhiều hơn cho giờ cao điểm) hoặc GPU. Nếu tài nguyên hạn chế, BoW + MLP (p95 0.21 ms, F1 0.915) là phương án dự phòng rẻ. LLM bị loại: p95 1.4–2 s vượt SLA ~30–40 lần, và theo phép tính tay ($10.5/ngày cho 1M tin) thì 10M tin tốn ~$105/ngày ≈ **$3.150/tháng**. LLM chỉ nên dùng ở tầng router cho tin lưỡng lự (p ∈ [0.3, 0.7]), xử lý bất đồng bộ.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời/viết gì | Tôi sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 4 (4.1–4.4) | _(điền prompt)_ | _(điền)_ | Chạy bộ test giả lập (✅ 4.1 + 4.4, 4.2–4.3) |
| Báo cáo | Tóm tắt kết quả từ notebook + `lab1_report.json` thành báo cáo này | Bản nháp các mục 2–6 | Đối chiếu từng số với output notebook |

**Điều học được:** Điểm cao trên test cùng phân phối không chứng minh model hiểu bài toán — BoW từ F1 0.92 rơi xuống 0.29 khi gặp spam viết tự nhiên. Chất lượng phải đặt cạnh độ trễ và chi phí: phương án "thông minh" nhất (LLM) lại không qua được SLA. Và pipeline có thể hỏng âm thầm (40/40 invalid vẫn cho ra một con số) — luôn kiểm tra output thô trước khi tin vào metric.
