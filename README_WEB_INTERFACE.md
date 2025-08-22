# Patent AI Agent - Web Interface

Giao diện web cho hệ thống trích xuất từ khóa patent với human-in-the-loop evaluation, thay thế phương thức CLI truyền thống.

## 🚀 Tính năng chính

- **Giao diện thân thiện**: Web interface đơn giản, tinh gọn và dễ sử dụng
- **Human-in-the-loop**: Cho phép người dùng đánh giá và chỉnh sửa kết quả trích xuất
- **Real-time processing**: Sử dụng WebSocket để cập nhật trạng thái real-time
- **Kết quả chi tiết**: Hiển thị đầy đủ concept matrix, keywords, IPC classifications, và patent URLs

## 📋 Yêu cầu hệ thống

- Node.js (v14 hoặc cao hơn)
- Python 3.8+
- Tất cả dependencies trong `requirements.txt`

## 🛠️ Cài đặt

### 1. Cài đặt Node.js dependencies

```bash
npm install
```

### 2. Cài đặt Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Cấu hình API keys

Đảm bảo các API keys trong `config/settings.py` đã được cấu hình đúng:
- TAVILY_API_KEY
- BRAVE_API_KEY
- OpenAI API key (trong extractor.py)

## 🏃‍♂️ Chạy ứng dụng

### Khởi động server

```bash
npm start
```

Hoặc để development với auto-reload:

```bash
npm run dev
```

Server sẽ chạy tại: http://localhost:3000

## 📱 Cách sử dụng

### 1. Nhập thông tin
- **Problem Description**: Mô tả vấn đề kỹ thuật mà phát minh muốn giải quyết
- **Technical Content**: Mô tả các khía cạnh kỹ thuật, phương pháp hoặc hệ thống

### 2. Xem kết quả trích xuất
Hệ thống sẽ hiển thị:
- **Concept Matrix**: Ma trận khái niệm chính
- **Seed Keywords**: Từ khóa gốc theo từng danh mục
- **IPC Classifications**: Phân loại IPC tự động

### 3. Human Evaluation
Chọn một trong các tùy chọn:
- **✅ Approve**: Chấp nhận kết quả hiện tại
- **❌ Reject**: Từ chối và tái tạo với feedback
- **✏️ Edit**: Chỉnh sửa thủ công (sẽ được triển khai)

### 4. Xem kết quả cuối cùng
- **Expanded Keywords**: Từ khóa được mở rộng với synonyms
- **Search Queries**: Các câu truy vấn tìm kiếm được tạo
- **Related Patents**: Danh sách patent liên quan với điểm số
- **Summary Statistics**: Thống kê tổng quan

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐    HTTP/WebSocket    ┌─────────────────┐
│   Frontend      │ ◄─────────────────► │   Node.js       │
│   (HTML/CSS/JS) │                     │   Server        │
└─────────────────┘                     └─────────────────┘
                                                │
                                        Process Communication
                                                │
                                        ┌─────────────────┐
                                        │   Python        │
                                        │   Backend       │
                                        │   (Extractor)   │
                                        └─────────────────┘
```

### Các file chính:

- **`server.js`**: Node.js server với Express và Socket.IO
- **`web_extractor.py`**: Python wrapper cho giao tiếp với Node.js
- **`public/index.html`**: Frontend interface
- **`src/core/extractor.py`**: Core extraction logic (original)

## 🔧 So sánh với CLI

| Tính năng | CLI (Original) | Web Interface |
|-----------|----------------|---------------|
| Input method | Terminal input | Web form |
| Human evaluation | CLI prompts | Web buttons |
| Results display | Text output | Rich HTML |
| Real-time updates | No | Yes (WebSocket) |
| User experience | Technical | User-friendly |
| Multi-session | No | Yes |

## 🐛 Xử lý lỗi

### Lỗi thường gặp:

1. **"Both problem and technical fields are required"**
   - Đảm bảo cả hai trường đều được điền

2. **"Python process exited unexpectedly"**
   - Kiểm tra Python dependencies
   - Xem log trong `web_extraction.log`

3. **"Network error"**
   - Kiểm tra kết nối mạng
   - Xác thực API keys

### Debug:

```bash
# Xem Python logs
tail -f web_extraction.log

# Xem Node.js logs
# Logs sẽ hiển thị trong terminal chạy server
```

## 🔮 Tính năng tương lai

- [ ] Manual keyword editing interface
- [ ] Export results to various formats (JSON, CSV, PDF)
- [ ] User authentication and session management
- [ ] Batch processing multiple inputs
- [ ] Advanced filtering and sorting for results
- [ ] Integration with patent databases

## 🤝 Đóng góp

Để đóng góp vào project:

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

[Add your license information here]

## 📞 Hỗ trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra logs
2. Xem phần troubleshooting
3. Tạo issue trên GitHub repository

---

**Lưu ý**: Web interface này hoàn toàn thay thế được phương thức CLI truyền thống, cung cấp trải nghiệm người dùng tốt hơn nhiều với giao diện trực quan và tương tác real-time.
