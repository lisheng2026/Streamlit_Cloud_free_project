import whisper
import os
import subprocess
import sys
from datetime import timedelta
import streamlit as st
import tempfile

# ===================== 自动安装依赖 =====================
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = {
    "whisper": "openai-whisper",
    "torch": "torch",
    "ffmpeg": "ffmpeg-python",
    "streamlit": "streamlit"
}

for pkg_import, pkg_install in required_packages.items():
    try:
        __import__(pkg_import)
    except ImportError:
        st.info(f"正在安装 {pkg_install} 依赖...")
        install_package(pkg_install)

import whisper
import ffmpeg

# ===================== 核心工具函数 =====================
def format_time(seconds):
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = td.total_seconds() % 60
    milliseconds = int((secs - int(secs)) * 1000)
    secs = int(secs)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def extract_raw_audio(video_path, audio_output_path):
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_output_path, format="wav")
            .overwrite_output()
            .run(quiet=True, capture_stderr=True)
        )
        return os.path.abspath(audio_output_path)
    except Exception as e:
        st.error(f"音频提取失败：{str(e)}")
        raise

def generate_srt_file(segments, output_srt_path):
    srt_content = ""
    for idx, seg in enumerate(segments, 1):
        start_time = format_time(seg["start"])
        end_time = format_time(seg["end"])
        text = seg["text"].strip()
        srt_content += f"{idx}\n{start_time} --> {end_time}\n{text}\n\n"
    
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return os.path.abspath(output_srt_path)

def process_video(video_file, model_size, language):
    """处理上传的视频文件"""
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 保存上传的视频文件
        video_path = os.path.join(tmp_dir, video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        
        # 定义输出路径
        audio_output = os.path.join(tmp_dir, "extracted_audio.wav")
        original_sub = os.path.join(tmp_dir, "original_language.srt")
        english_sub = os.path.join(tmp_dir, "english_translated.srt")

        # 1. 提取音频
        st.info("📌 开始提取视频音频...")
        audio_path = extract_raw_audio(video_path, audio_output)
        st.success(f"✅ 音频提取完成：{audio_path}")

        # 2. 加载模型
        st.info(f"📌 加载Whisper {model_size} 模型（首次运行需下载~4GB）...")
        model = whisper.load_model(model_size)

        # 3. 识别原语言
        st.info("📌 开始语音识别...")
        transcribe_result = model.transcribe(
            video_path,
            language=language,
            verbose=False,
            fp16=False
        )
        detected_lang = transcribe_result["language"]
        st.success(f"✅ 自动识别语言：{detected_lang}")

        # 4. 生成原语言字幕
        st.info("📌 生成原语言字幕...")
        original_sub_path = generate_srt_file(transcribe_result["segments"], original_sub)
        st.success(f"✅ 原语言字幕生成完成：{original_sub_path}")

        # 5. 生成英文翻译字幕
        st.info("📌 生成英文翻译字幕...")
        if detected_lang != "en":
            translate_result = model.transcribe(
                video_path,
                task="translate",
                language=detected_lang,
                verbose=False,
                fp16=False
            )
            english_sub_path = generate_srt_file(translate_result["segments"], english_sub)
        else:
            st.info("ℹ️ 原语言为英语，直接复制原字幕至英文文件")
            english_sub_path = generate_srt_file(transcribe_result["segments"], english_sub)
        st.success(f"✅ 英文字幕生成完成：{english_sub_path}")

        # 返回文件供下载
        return {
            "audio": (audio_output, "extracted_audio.wav"),
            "original_sub": (original_sub, "original_language.srt"),
            "english_sub": (english_sub, "english_translated.srt")
        }

# ===================== Streamlit 界面 =====================
def main():
    st.set_page_config(
        page_title="Whisper 字幕生成工具",
        page_icon="🎬",
        layout="wide"
    )

    st.title("🎬 Whisper 视频字幕生成工具")
    st.markdown("基于OpenAI Whisper的多语言视频字幕生成，支持自动识别语言+翻译为英文")

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置项")
        model_size = st.selectbox(
            "模型大小（速度/精度平衡）",
            options=["base", "small", "medium", "large"],
            index=2,
            help="base最快（~1GB），large最准（~6GB）"
        )
        language = st.selectbox(
            "识别语言",
            options=["自动识别", "zh", "en", "nan", "vi", "ja", "ko"],
            index=0,
            help="指定语言可提高识别准确率"
        )
        language = None if language == "自动识别" else language

    # 主界面文件上传
    st.header("📤 上传视频文件")
    video_file = st.file_uploader(
        "选择视频文件（支持MP4/AVI/MKV等）",
        type=["mp4", "avi", "mkv", "mov", "flv", "wmv"]
    )

    # 处理按钮
    if st.button("🚀 开始生成字幕", type="primary", disabled=not video_file):
        with st.spinner("处理中，请耐心等待（模型加载/识别可能需要几分钟）..."):
            try:
                # 处理视频
                result_files = process_video(video_file, model_size, language)
                
                # 显示下载区域
                st.header("📥 下载生成的文件")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    with open(result_files["audio"][0], "rb") as f:
                        st.download_button(
                            label="下载提取的音频",
                            data=f,
                            file_name=result_files["audio"][1],
                            mime="audio/wav"
                        )
                
                with col2:
                    with open(result_files["original_sub"][0], "rb") as f:
                        st.download_button(
                            label="下载原语言字幕",
                            data=f,
                            file_name=result_files["original_sub"][1],
                            mime="text/plain"
                        )
                
                with col3:
                    with open(result_files["english_sub"][0], "rb") as f:
                        st.download_button(
                            label="下载英文翻译字幕",
                            data=f,
                            file_name=result_files["english_sub"][1],
                            mime="text/plain"
                        )
                
                st.success("🎉 所有文件生成完成！")

            except Exception as e:
                st.error(f"❌ 处理失败：{str(e)}")

    # 说明文档
    with st.expander("📖 使用说明"):
        st.markdown("""
        1. 上传视频文件（支持MP4/AVI/MKV等常见格式）；
        2. 选择模型大小（推荐medium，平衡速度和精度）；
        3. 选择识别语言（自动识别即可，也可手动指定）；
        4. 点击「开始生成字幕」，等待处理完成；
        5. 下载生成的音频文件、原语言字幕、英文翻译字幕。
        
        注意：首次运行会自动下载Whisper模型（medium约4GB），请确保网络畅通。
        """)

if __name__ == "__main__":
    main()
