import streamlit as st
from models import SegDecNet
import torch
import cv2
import numpy as np
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT_WIDTH = 512  # must be the same as it was during training
INPUT_HEIGHT = 512  # must be the same as it was during training
INPUT_CHANNELS = 1  # must be the same as it was during training
dsize = INPUT_WIDTH, INPUT_HEIGHT
device = "cpu"  # cpu or cuda:IX

def load_image(uploaded_file):
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        img = cv2.imdecode(np.frombuffer(io.BytesIO(image_data).read(), np.uint8), 0)
        img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
        # st.write(img.shape)
        img = np.transpose(img, (2, 0, 1)) if INPUT_CHANNELS == 3 else img[np.newaxis]
        img_t = torch.from_numpy(img)[np.newaxis].float() / 255.0  # must be [BATCH_SIZE x CHANNELS x HEIGHT x WIDTH]
        # st.write(img_t.shape)
        return img_t
    else:
        return None

def load_model(option):
    model = SegDecNet(device, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)
    model.set_gradient_multipliers(0)
    fold = option.split('-')[-1]
    model_path = r"D:\\Chonnam\\Semester3\\ProjectForAI\\mixed-segdec-net-comind2021\\results\\DAGM\\N_ALL\\FOLD_"+ fold +"\\models\\final_state_dict.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def plot_sample(segmentation, decision=None):
    jet_seg = (segmentation / segmentation.max() * 255).astype(np.uint8)
    jet_seg = cv2.applyColorMap((segmentation * 255).astype(np.uint8), cv2.COLORMAP_JET)
    jet_seg = cv2.putText(jet_seg, f"Score: {decision:.3f}", (2, 25),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),3)
    return jet_seg
    
def predict(model, uploaded_file, image):    
    if uploaded_file is not None:
        dec_out, seg_out = model(image)
        pred_seg = torch.sigmoid(seg_out)
        prediction = torch.sigmoid(dec_out)
        prediction = prediction.item()
        pred_seg = pred_seg.detach().cpu().numpy()
        pred_seg = cv2.resize(pred_seg[0, 0, :, :], dsize) if len(pred_seg.shape) == 4 else cv2.resize(pred_seg[0, :, :], dsize)
        pred_mask = plot_sample(segmentation=pred_seg, decision=prediction)
        st.image(pred_mask, caption=None, clamp=True, channels='BGR')

def main():
    st.title('AI-based Part Defect Detection')
    page = st.sidebar.selectbox('Page Navigation', ["Predictor", "Model analysis"])
    st.sidebar.markdown("""---""")
    st.sidebar.write("Created by [LHD Pte Ltd](https://www.facebook.com/lehoaiduy1396/)")
    if page == "Predictor":
        st.markdown("Select input image")
        option = st.selectbox('What class you want to run?', 
                             ('DAGM-1', 'DAGM-2', 'DAGM-3', 'DAGM-4','DAGM-5',
                             'DAGM-6', 'DAGM-7','DAGM-8', 'DAGM-9','DAGM-10'))
        model = load_model(option) 
        uploaded_file = st.file_uploader(label="Choose a image file")
        result = st.button('Predict')
        col1, col2 = st.columns(2)
        with col1:
            st.header("Input")
            image = load_image(uploaded_file)             
        with col2:
            st.header("Output")          
            if result:
                predict(model, uploaded_file, image)

if __name__ == '__main__':
    main()