import streamlit as st
import numpy as np
import joblib

from PIL import Image

# تحميل الموديل

model = joblib.load("model.pkl")
st.title("🖊 Handwritten Digit Recognition (MNIST)")

st.write("ارفع صورة رقم (28x28) أبيض وأسود علشان الموديل يتنبأ")

# رفع صورة
uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # عرض الصورة
    image = Image.open(uploaded_file).convert("L").resize((28,28))
    st.image(image, caption="الصورة المرفوعة", width=150)

    # تجهيز الصورة
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28*28)

    # تنبؤ
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction, axis=1)[0]


    st.success(f"📌 الرقم المتوقع هو: {pred_class}")
