# app.py
import streamlit as st
import numpy as np
import pickle

# Load mô hình và scaler
model = pickle.load(open("logistic_asd_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🔎 Ứng dụng Sàng Lọc Tự Kỷ (ASD)")

# Giao diện người dùng
st.header("📝 Vui lòng nhập thông tin:")

# A1 đến A10
scores = [st.slider(f"Câu hỏi A{i+1}", 0, 1, 0, help="0 = Không, 1 = Có") for i in range(10)]

# Các đặc trưng khác
used_app_before = st.selectbox("Bạn đã từng sử dụng ứng dụng này chưa?", ["Chưa", "Rồi"])
jundice = st.selectbox("Trẻ có bị vàng da sau sinh không?", ["Không", "Có"])
austim = st.selectbox("Gia đình bạn có người từng bị tự kỷ không?", ["Không", "Có"])
relation = st.selectbox("Bạn là ai đối với người được đánh giá?", ["Bố/mẹ", "Bản thân", "Người thân khác", "Khác"])
age_desc = st.selectbox("Nhóm tuổi của người được đánh giá?", ["Dưới 18 tuổi", "Từ 18 tuổi trở lên"])

# Mã hóa dữ liệu đầu vào giống như mô hình đã huấn luyện
used_app_before_val = 1 if used_app_before == "Rồi" else 0
jundice_val = 1 if jundice == "Có" else 0
austim_val = 1 if austim == "Có" else 0

relation_map = {"Bố/mẹ": 0, "Bản thân": 1, "Người thân khác": 2, "Khác": 3}
relation_val = relation_map[relation]

age_desc_map = {"Dưới 18 tuổi": 0, "Từ 18 tuổi trở lên": 1}
age_desc_val = age_desc_map[age_desc]

# Tạo vector đầu vào
X_input = np.array(scores + [used_app_before_val, jundice_val, relation_val, austim_val, age_desc_val]).reshape(1, -1)

# Chuẩn hóa
X_input_scaled = scaler.transform(X_input)

# Dự đoán
if st.button("🔍 Dự đoán"):
    pred = model.predict(X_input_scaled)[0]

    st.markdown("### 📋 Kết quả sàng lọc:")
    if pred == 1:
        st.warning("🔔 Có thể có dấu hiệu tự kỷ. Bạn nên tham khảo ý kiến chuyên gia.")
        st.markdown("""
        ### 🧭 Gợi ý hành động tiếp theo:
        🔹 Hãy liên hệ chuyên gia tâm lý hoặc cơ sở y tế để được tư vấn kỹ lưỡng hơn.  
        🔹 Ghi chép lại các biểu hiện thường gặp trong cuộc sống hàng ngày.  
        🔹 Tham khảo tài liệu từ WHO, CDC hoặc các trung tâm hỗ trợ trong nước.  
        """)
    else:
        st.success("✅ Không có dấu hiệu rõ ràng của tự kỷ tại thời điểm này.")
        st.markdown("""
        ### 🧭 Gợi ý tiếp theo:
        🔹 Hãy tiếp tục theo dõi và hỗ trợ phát triển kỹ năng xã hội, cảm xúc cho trẻ.  
        🔹 Nếu còn băn khoăn, bạn có thể trao đổi thêm với chuyên gia.  
        """)
