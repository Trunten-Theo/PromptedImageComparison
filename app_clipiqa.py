# This file uses code from CLIP-IQA (https://github.com/IceClear/CLIP-IQA)
# Licensed under the Apache License, Version 2.0.

import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
from torchvision import transforms
from torchmetrics.multimodal import CLIPImageQualityAssessment

# 画像前処理
def trans_image(image):

    # 画像をテンソルに変換するためのトランスフォームを定義
    transform = transforms.Compose([
        transforms.Resize((224, 224)),    # 画像を224x224にリサイズ
        transforms.ToTensor(),            # テンソルに変換
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
    ])

    # 画像をテンソルに変換
    img = transform(image).unsqueeze(0)

    return img

# 初期化
def clear_all():
    st.session_state.val_prompts_checkbox = []
    st.session_state.val_prompts_custom = []
    st.session_state.pprompt = ""
    st.session_state.nprompt = ""
    for label in options:
        st.session_state[f"checkbox_{label}"] = False

# 結果バー出力
def result(index, value):
    left_label = val_prompts[index][0]
    right_label = val_prompts[index][1]

    # ラベルスタイルをvalueに応じて切り替える
    if value > 0.5:
        left_style = "font-weight: bold; background-color: gold; padding: 2px; border-radius: 3px;"
        right_style = "color: black;"
    else:
        left_style = "color: black;"
        right_style = "font-weight: bold; background-color: gold; padding: 2px; border-radius: 3px;"

    st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 10px;">
                <!-- 左ラベル -->
                <div style="width: 100px; text-align: right; {left_style}">{left_label}</div>
                <!-- カラーバー -->
                <div style="flex-grow: 1; height: 30px; position: relative;
                background: linear-gradient(to right, lightgreen 0%, lightgreen {value*100}%, red {value*100}%, red 100%);
                border: 1px solid #ccc; border-radius: 5px;">
                <!-- 左スコア -->
                <div style="position: absolute; left: 10px; top: 0; bottom: 0; display: flex; align-items: center; font-weight: bold; color: black;">
                {int(value * 100)}%
                </div>
                <!-- 右スコア -->
                <div style="position: absolute; right: 10px; top: 0; bottom: 0; display: flex; align-items: center; font-weight: bold; color: white;">
                {int((1 - value) * 100)}%
                </div>
                </div>
                <!-- 右ラベル -->
                <div style="width: 100px; text-align: left; {right_style}">{right_label}</div>
                </div>
                """, unsafe_allow_html=True)

# 画面UI
st.set_page_config(layout="wide")  # 横幅を広く使えるように
st.title("画像内の似ている特徴を判別するやつ")
st.markdown("""
似ている特徴を<a href='https://arxiv.org/pdf/2207.12396'>CLIP-IQA</a>で判別する。
""", unsafe_allow_html=True)


# 横2列に分割
col1, col2 = st.columns([1, 1])  # [左, 右]の幅比を指定可能

with col1:
    # st.header("① 画像選択")
    st.markdown("""
<div style='background-color: #f0f8ff; padding: 10px 15px; border-radius: 5px; font-size: 24px; font-weight: bold;'>
1️⃣ 画像選択
</div>
""", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="アップロードされた画像", use_container_width=False, width=600)

with col2:
    st.markdown("""
<div style='background-color: #f0f8ff; padding: 10px 15px; border-radius: 5px; font-size: 24px; font-weight: bold;'>
2️⃣ モデル選択
</div>
""", unsafe_allow_html=True)
    options = ["clip_iqa", "openai/clip-vit-base-patch16", "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14-336", "openai/clip-vit-large-patch14"]
    selected = st.selectbox("モデル", options, index=0)

    # st.header("② プロンプト選択")
    st.markdown("""
<div style='background-color: #f0f8ff; padding: 10px 15px; border-radius: 5px; font-size: 24px; font-weight: bold;'>
3️⃣ プロンプト選択
</div>
""", unsafe_allow_html=True)

    # セッション状態で両方保持
    if "val_prompts_checkbox" not in st.session_state:
        st.session_state.val_prompts_checkbox = []

    if "val_prompts_custom" not in st.session_state:
        st.session_state.val_prompts_custom = []

    options = {
    "青空判定": ("blue sky", "gray sky"),
    "タンクレストイレ判定": ("toilet", "tankless-toilet"),
    "3口コンロ判定": ("three", "two")
    }

    val_prompts_default = []

    # 固定プロンプト
    st.markdown("##### ・固定プロンプト")
    for label, pair in options.items():
        checked = st.checkbox(label, key=f"checkbox_{label}")

        if checked:
            if pair not in st.session_state.val_prompts_checkbox:
                st.session_state.val_prompts_checkbox.append(pair)
        else:
            if pair in st.session_state.val_prompts_checkbox:
                st.session_state.val_prompts_checkbox.remove(pair)


    # ユーザ指定プロンプト
    st.markdown("##### ・ユーザ指定プロンプト")
    col1, col2, col3 = st.columns([3, 3, 1])
    with col1:
        prompt_positive = st.text_input("ラベル1", key="pprompt")
    with col2:
        prompt_negative = st.text_input("ラベル2", key="nprompt")
    with col3:
        # 追加ボタン
        add_button = st.button("追加")
        # 設定初期化ボタン
        st.button("クリア", on_click=clear_all)

    # 追加処理
    if add_button and prompt_positive and prompt_negative:
        new_pair = (prompt_positive, prompt_negative)
        val = st.session_state.pprompt
        if new_pair not in st.session_state.val_prompts_custom:
            st.session_state.val_prompts_custom.append(new_pair)
    # st.write("📝 現在の val_prompts:", st.session_state.val_prompts)

    # 2つのリストを結合
    ## CLIPIQAへの入力がタプルのため変換
    val_prompts = tuple(
        st.session_state.val_prompts_checkbox + st.session_state.val_prompts_custom
    )
    st.write("🧩 分類対象プロンプト:", val_prompts)


    if uploaded_file:
        # st.header("③ スコア比較")
        st.markdown("""
<div style='background-color: #f0f8ff; padding: 10px 15px; border-radius: 5px; font-size: 24px; font-weight: bold;'>
4️⃣ スコア比較
</div>
""", unsafe_allow_html=True)
        if st.button("実行"):

            # CLIPモデル
            # clip_model = "clip_iqa"
            clip_model = selected
            # ”clip_iqa” デフォルト、CLIP-IQAのスコア
            # ”openai/clip-vit-base-patch16”
            # ”openai/clip-vit-base-patch32”
            # ”openai/clip-vit-large-patch14-336”
            # ”openai/clip-vit-large-patch14”

            image = trans_image(image)

            clipiqa_scores = []
            metric = CLIPImageQualityAssessment(clip_model, prompts=val_prompts)

            score_output = metric(image)

            # プロンプトが複数の場合
            if isinstance(score_output, dict):
                for index, (key, value) in enumerate(score_output.items()):
                    # print(f"{key}: {round(value.item(), 3)}")
                    result(index, value)

            # プロンプトが1つの場合
            else:
                index = 0
                value = score_output.item()
                result(index, value)
                # print(f"{val_prompts[0][0]}: {round(score_output.item(), 3)}")

            # print(clipiqa_scores)

st.markdown("""
<br><br><br>
<div style='text-align: center; color: gray; font-size: 14px;'>
  © 2025 服部翔. All rights reserved.<br>
This application uses <a href='https://github.com/IceClear/CLIP-IQA' target='_blank'>CLIP-IQA</a>
  under the Apache 2.0 License.
</div>
""", unsafe_allow_html=True)
