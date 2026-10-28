import warnings
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
import streamlit as st

# 🚫 徹底關閉所有警告
warnings.filterwarnings('ignore')

# ==========================================
# 0. Streamlit 頁面基本設定
# ==========================================
st.set_page_config(
    page_title='Sustainable Textile Sorting System', page_icon='🌿', layout='centered'
)

st.title('🌿 AI-Powered Sustainable Textile Classification System')
st.markdown(
    'Upload fabric images to classify material and dispatch to sustainable'
    ' recycling zones.'
)


# ==========================================
# 1. 載入訓練好的模型與 Metadata (使用快取加速)
# ==========================================
@st.cache_resource
def load_model():
  model_path = 'fabric_lbp_rf_9classes.pkl'  # 確保模型檔案與 app.py 在同一目錄
  saved_data = joblib.load(model_path)
  return saved_data['model'], saved_data['classes']


try:
  clf, target_classes = load_model()
  st.sidebar.success('✅ Model loaded successfully!')
except Exception as e:
  st.sidebar.error(
      f'⚠️ Failed to load model. Please ensure the pkl file exists. Error: {e}'
  )


# ==========================================
# 2. 影像預處理與 LBP 特徵提取函數
# ==========================================
def preprocess_image(img_gray):
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  img_clahe = clahe.apply(img_gray)
  blur = cv2.GaussianBlur(img_clahe, (21, 21), 0)
  return cv2.addWeighted(img_clahe, 1.5, blur, -0.5, 0)


def extract_lbp_features(img_gray, radius=3, n_points=24):
  lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
  hist, _ = np.histogram(
      lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
  )
  hist = hist.astype('float')
  hist /= hist.sum() + 1e-6  # L1 正規化
  return hist


# ==========================================
# 3. 側邊欄與檔案上傳介面
# ==========================================
st.sidebar.header('Upload Section')
uploaded_files = st.sidebar.file_uploader(
    'Choose fabric images...', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True
)

if uploaded_files:
  st.markdown('---')
  st.subheader('🔍 Analysis Results')

  for uploaded_file in uploaded_files:
    # 讀取圖片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
      st.error(f'Could not read {uploaded_file.name}.')
      continue

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 縮放、預處理與特徵提取
    img_resized = cv2.resize(img_gray, (256, 256))
    processed_img = preprocess_image(img_resized)
    lbp_feat = extract_lbp_features(processed_img).reshape(1, -1)

    # 取得所有類別的預測機率
    pred_proba = clf.predict_proba(lbp_feat)[0]

    # 計算三大專區可能性之和
    zone_probs = {
        '1) Natural Fiber Processing': 0.0,
        '2) Synthetic Chemical Granulation': 0.0,
        '3) High-Value Second-Hand & Upcycling': 0.0,
    }

    natural_zone = ['Cotton', 'Silk', 'train_linen', 'Wool', 'Denim']
    synthetic_zone = ['Acrylic', 'Nylon', 'Polyester']
    leather_zone = ['Leather']

    for idx, prob in enumerate(pred_proba):
      c_name = target_classes[idx]
      if c_name in natural_zone:
        zone_probs['1) Natural Fiber Processing'] += prob
      elif c_name in synthetic_zone:
        zone_probs['2) Synthetic Chemical Granulation'] += prob
      elif c_name in leather_zone:
        zone_probs['3) High-Value Second-Hand & Upcycling'] += prob

    # 取得最高信心材質與最高機率分區
    top1_idx = np.argmax(pred_proba)
    top1_class = target_classes[top1_idx]
    top1_conf = pred_proba[top1_idx] * 100

    top_zone = max(zone_probs, key=zone_probs.get)
    top_zone_conf = zone_probs[top_zone] * 100

    # 畫面左右排版：左邊放圖片，右邊放分析數據
    col1, col2 = st.columns([1, 2])

    with col1:
      st.image(
          cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
          caption=uploaded_file.name,
          use_container_width=True,
      )

    with col2:
      st.markdown(f'### **File:** {uploaded_file.name}')
      st.markdown(
          f'🎯 **Top Material Prediction:** `{top1_class}` (`{top1_conf:.2f}%`'
          ' confidence)'
      )
      st.markdown(
          f'🏭 **Assigned Recycling Zone:** `{top_zone}` (`{top_zone_conf:.2f}%`'
          ' total sum)'
      )

      # 顯示各區機率細節
      with st.expander('View All Recycling Zone Probabilities'):
        for z_name, z_prob in zone_probs.items():
          st.progress(
              float(z_prob), text=f'{z_name}: {z_prob * 100:.2f}%'
          )

    st.markdown('---')
else:
  st.info('👈 Please upload one or more fabric images from the sidebar to begin.')
