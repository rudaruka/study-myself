
# ============================================================
#  PresentIQ — 발표 능력 분석 도우미 (Streamlit 버전)
#
#  실행: streamlit run app.py
#  배포: GitHub → Streamlit Cloud (share.streamlit.io)
#
#  필요 라이브러리: requirements.txt 참고
# ============================================================

import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import detect_silence
import librosa
import numpy as np
import tempfile
import os
import io
import plotly.graph_objects as go

# ── 마이크 녹음 위젯 (audio-recorder-streamlit) ──────────────
try:
    from audio_recorder_streamlit import audio_recorder
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PresentIQ — 발표 분석 도우미",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# 커스텀 CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

/* 전체 배경 */
[data-testid="stAppViewContainer"] {
    background: #080d14;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="block-container"] { padding-top: 2rem; }

/* 제목 */
.piq-header {
    text-align: center;
    padding: 10px 0 36px;
}
.piq-badge {
    display: inline-block;
    background: rgba(0,212,170,.12);
    border: 1px solid rgba(0,212,170,.3);
    border-radius: 50px;
    padding: 5px 18px;
    font-family: 'Space Mono', monospace;
    font-size: .72rem;
    letter-spacing: .15em;
    color: #00d4aa;
    margin-bottom: 18px;
}
.piq-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(120deg,#fff 30%,#00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 10px;
}
.piq-sub {
    color: #5a7a99;
    font-size: .98rem;
}

/* 섹션 카드 */
.piq-card {
    background: #0f1923;
    border: 1px solid #1e3048;
    border-radius: 14px;
    padding: 26px 28px;
    margin-bottom: 20px;
}
.piq-section-label {
    font-family: 'Space Mono', monospace;
    font-size: .65rem;
    letter-spacing: .18em;
    color: #00d4aa;
    text-transform: uppercase;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e3048;
}

/* 지표 카드 */
.metric-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
.metric-box {
    flex: 1; min-width: 130px;
    background: #162030;
    border: 1px solid #1e3048;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    position: relative; overflow: hidden;
}
.metric-box::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg,#00d4aa,#0098d4);
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #00d4aa;
    line-height: 1.1;
}
.metric-lbl {
    font-size: .72rem; color: #5a7a99;
    margin-top: 5px; text-transform: uppercase; letter-spacing: .06em;
}

/* 피드백 */
.fb { display:flex; gap:14px; align-items:flex-start;
      padding:14px 18px; border-radius:12px; margin-bottom:10px;
      font-size:.9rem; line-height:1.6; }
.fb.good    { background:rgba(61,220,132,.08);  border:1px solid rgba(61,220,132,.25); }
.fb.warning { background:rgba(255,107,107,.08); border:1px solid rgba(255,107,107,.25); }
.fb.caution { background:rgba(255,209,102,.08); border:1px solid rgba(255,209,102,.25); }
.fb-cat { font-weight:700; font-size:.8rem; min-width:110px; white-space:nowrap; }
.fb.good    .fb-cat { color:#3ddc84; }
.fb.warning .fb-cat { color:#ff6b6b; }
.fb.caution .fb-cat { color:#ffd166; }
.fb-msg { color:#dce8f5; }

/* 텍스트 박스 */
.transcript {
    background: #162030; border:1px solid #1e3048; border-radius:10px;
    padding:18px 20px; font-size:.9rem; line-height:1.85;
    color:#dce8f5; min-height:60px; white-space:pre-wrap;
}

/* 예시 코드 블록 */
.example {
    background: #162030; border-left: 3px solid #0098d4;
    border-radius: 0 10px 10px 0;
    padding: 18px 22px;
    font-family: 'Space Mono', monospace;
    font-size: .76rem; line-height:2; color:#8ab4cc;
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# [함수 1] 오디오 → WAV 변환
# ─────────────────────────────────────────────────────────────
def convert_to_wav(src_path: str) -> str:
    """다양한 오디오 형식을 WAV로 변환한다 (pydub 사용)."""
    audio = AudioSegment.from_file(src_path)
    wav_path = src_path.rsplit('.', 1)[0] + '_conv.wav'
    audio.export(wav_path, format='wav')
    return wav_path


# ─────────────────────────────────────────────────────────────
# [함수 2] 음성 → 텍스트 변환 (STT)
# ─────────────────────────────────────────────────────────────
def transcribe_audio(wav_path: str) -> str:
    """
    SpeechRecognition + Google Web Speech API로
    음성을 한국어 텍스트로 변환한다.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data, language='ko-KR')
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        raise ConnectionError(f"음성 인식 API 오류: {e}")


# ─────────────────────────────────────────────────────────────
# [함수 3] 단어 수 계산 (리스트 활용)
# ─────────────────────────────────────────────────────────────
def count_words(text: str) -> int:
    """한국어 텍스트를 공백으로 분리해 어절 수를 계산한다."""
    if not text:
        return 0
    word_list = text.split()  # 리스트 생성
    return len(word_list)


# ─────────────────────────────────────────────────────────────
# [함수 4] 말 속도 계산 (WPM)
# ─────────────────────────────────────────────────────────────
def calculate_wpm(word_count: int, duration_sec: float) -> float:
    """분당 단어 수(WPM) = 단어 수 / 발표 시간(분)."""
    if duration_sec <= 0:
        return 0.0
    return round(word_count / (duration_sec / 60), 1)


# ─────────────────────────────────────────────────────────────
# [함수 5] 침묵 구간 감지 (for 반복문 활용)
# ─────────────────────────────────────────────────────────────
def detect_silences(wav_path: str,
                    min_len: int = 1000,
                    thresh: int = -40) -> list:
    """
    1초 이상 조용한 구간을 감지한다.
    결과를 리스트로 반환한다.
    """
    audio = AudioSegment.from_wav(wav_path)
    raw   = detect_silence(audio, min_silence_len=min_len, silence_thresh=thresh)

    silence_list = []  # 리스트: 침묵 구간 정보 저장
    for start_ms, end_ms in raw:              # for 반복문
        silence_list.append({
            'start':    round(start_ms / 1000, 1),
            'end':      round(end_ms   / 1000, 1),
            'duration': round((end_ms - start_ms) / 1000, 1),
        })
    return silence_list


# ─────────────────────────────────────────────────────────────
# [함수 6] 오디오 길이 계산
# ─────────────────────────────────────────────────────────────
def get_duration(wav_path: str) -> float:
    """librosa로 오디오 총 길이(초)를 계산한다."""
    y, sr_rate = librosa.load(wav_path, sr=None)
    return round(librosa.get_duration(y=y, sr=sr_rate), 2)


# ─────────────────────────────────────────────────────────────
# [함수 7] 피드백 생성 (if 조건문 활용)
# ─────────────────────────────────────────────────────────────
def generate_feedback(wpm: float, silence_count: int, duration: float) -> list:
    """
    분석 결과를 바탕으로 맞춤형 피드백을 생성한다.
    if 조건문으로 각 지표를 판단하고 리스트로 반환한다.
    """
    feedbacks = []  # 리스트: 피드백 순서대로 저장

    # ── 말 속도 피드백 (한국어 발표 기준 80~180 WPM) ──
    if wpm > 180:
        feedbacks.append({'type': 'warning', 'cat': '🚀 말 속도',
            'msg': f'말 속도({wpm} WPM)가 너무 빠릅니다. 청중이 내용을 따라가기 어려울 수 있습니다. 천천히, 또렷하게 말하는 연습을 해보세요.'})
    elif 0 < wpm < 80:
        feedbacks.append({'type': 'warning', 'cat': '🐢 말 속도',
            'msg': f'말 속도({wpm} WPM)가 너무 느립니다. 청중의 집중력이 흐트러질 수 있습니다. 자신감 있게 말해보세요.'})
    elif wpm == 0:
        feedbacks.append({'type': 'warning', 'cat': '🎙️ 음성 인식',
            'msg': '음성이 인식되지 않았습니다. 마이크 상태나 발음을 확인해주세요.'})
    else:
        feedbacks.append({'type': 'good', 'cat': '✅ 말 속도',
            'msg': f'말 속도({wpm} WPM)가 적절합니다. 청중이 이해하기 좋은 속도를 유지하고 있습니다!'})

    # ── 침묵 구간 피드백 ──
    if silence_count > 5:
        feedbacks.append({'type': 'warning', 'cat': '⏸️ 침묵 구간',
            'msg': f'1초 이상 침묵이 {silence_count}회 감지되었습니다. 발표 내용을 충분히 숙지하고, "음…" 습관을 줄이는 연습을 해보세요.'})
    elif silence_count > 2:
        feedbacks.append({'type': 'caution', 'cat': '⚠️ 침묵 구간',
            'msg': f'침묵 구간이 {silence_count}회 감지되었습니다. 핵심 문장을 미리 암기하면 흐름이 더 자연스러워집니다.'})
    else:
        feedbacks.append({'type': 'good', 'cat': '✅ 발표 흐름',
            'msg': '침묵 구간이 거의 없어 발표 흐름이 매끄럽습니다!'})

    # ── 발표 시간 피드백 ──
    if duration < 30:
        feedbacks.append({'type': 'caution', 'cat': '⏱️ 발표 시간',
            'msg': '발표 시간이 30초 미만으로 짧습니다. 내용을 더 풍부하게 구성해보세요.'})
    elif duration > 600:
        feedbacks.append({'type': 'caution', 'cat': '⏱️ 발표 시간',
            'msg': '발표 시간이 10분을 초과합니다. 핵심 내용 위주로 간결하게 구성하면 더 효과적입니다.'})
    else:
        m, s = int(duration // 60), int(duration % 60)
        feedbacks.append({'type': 'good', 'cat': '✅ 발표 시간',
            'msg': f'발표 시간({m}분 {s}초)이 적절한 범위입니다!'})

    return feedbacks


# ─────────────────────────────────────────────────────────────
# [함수 8] 전체 분석 파이프라인
# ─────────────────────────────────────────────────────────────
def run_analysis(audio_bytes: bytes, ext: str = '.wav') -> dict:
    """
    바이트 데이터를 받아 전체 분석 과정을 수행하고 결과 딕셔너리를 반환한다.
    """
    tmp_path = wav_path = None
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        wav_path  = convert_to_wav(tmp_path)           # 1. WAV 변환
        duration  = get_duration(wav_path)              # 2. 총 길이
        text      = transcribe_audio(wav_path)          # 3. STT
        words     = count_words(text)                   # 4. 단어 수
        wpm       = calculate_wpm(words, duration)      # 5. WPM
        silences  = detect_silences(wav_path)           # 6. 침묵
        feedbacks = generate_feedback(wpm, len(silences), duration)  # 7. 피드백

        return {
            'duration':  duration,
            'text':      text or '(음성 인식 실패 — 발음이나 마이크를 확인해주세요)',
            'words':     words,
            'wpm':       wpm,
            'silences':  silences,
            'feedbacks': feedbacks,
        }
    finally:
        # 임시 파일 삭제
        for p in [tmp_path, wav_path]:
            if p and os.path.exists(p):
                os.unlink(p)


# ─────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────

# ── 헤더 ──
st.markdown("""
<div class="piq-header">
  <div class="piq-badge">● PRESENT IQ · 발표 분석 AI</div>
  <div class="piq-title">발표 능력 분석 도우미</div>
  <div class="piq-sub">음성을 업로드하거나 녹음하면 말 속도 · 침묵 · 흐름을 자동으로 분석합니다</div>
</div>
""", unsafe_allow_html=True)

# ── 입력 섹션 ──
st.markdown('<div class="piq-card"><div class="piq-section-label">음성 입력</div>', unsafe_allow_html=True)

audio_bytes = None  # 분석에 사용할 오디오 데이터
audio_ext   = '.wav'

tab_upload, tab_mic = st.tabs(["📁 파일 업로드", "🎙️ 마이크 녹음"])

with tab_upload:
    uploaded = st.file_uploader(
        "음성 파일을 선택하세요",
        type=['wav', 'mp3', 'm4a', 'ogg', 'webm', 'flac'],
        label_visibility='collapsed',
    )
    if uploaded:
        audio_bytes = uploaded.read()
        audio_ext   = os.path.splitext(uploaded.name)[1] or '.wav'
        st.audio(audio_bytes, format=f'audio/{audio_ext.lstrip(".")}')
        st.success(f"✅ 파일 업로드 완료: **{uploaded.name}**")

with tab_mic:
    if MIC_AVAILABLE:
        st.info("🎙️ 아래 버튼을 눌러 녹음을 시작·종료하세요.")
        rec_bytes = audio_recorder(
            text="",
            recording_color="#ff6b6b",
            neutral_color="#00d4aa",
            icon_size="2x",
        )
        if rec_bytes:
            audio_bytes = rec_bytes
            audio_ext   = '.wav'
            st.audio(rec_bytes, format='audio/wav')
            st.success("✅ 녹음 완료! 아래 버튼으로 분석을 시작하세요.")
    else:
        st.warning("`audio-recorder-streamlit` 패키지가 설치되어 있지 않습니다.  \n`pip install audio-recorder-streamlit` 후 재실행해주세요.")

st.markdown('</div>', unsafe_allow_html=True)

# ── 분석 버튼 ──
col_btn, _ = st.columns([1, 2])
with col_btn:
    analyze_clicked = st.button(
        "📊 발표 분석 시작",
        disabled=(audio_bytes is None),
        use_container_width=True,
        type='primary',
    )

# ─────────────────────────────────────────────────────────────
# 분석 실행 & 결과 표시
# ─────────────────────────────────────────────────────────────
if analyze_clicked and audio_bytes:
    with st.spinner("🔍 음성을 분석 중입니다… 잠시만 기다려주세요."):
        try:
            result = run_analysis(audio_bytes, audio_ext)
        except ConnectionError as e:
            st.error(f"❌ {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ 분석 중 오류가 발생했습니다: {e}")
            st.stop()

    st.balloons()

    # ── 수치 요약 ──
    dur = result['duration']
    m, s = int(dur // 60), int(dur % 60)
    dur_str = f"{m}:{str(s).zfill(2)}" if m else f"{s}s"

    st.markdown(f"""
    <div class="piq-card">
      <div class="piq-section-label">분석 결과 요약</div>
      <div class="metric-row">
        <div class="metric-box">
          <div class="metric-val">{dur_str}</div>
          <div class="metric-lbl">총 발표 시간</div>
        </div>
        <div class="metric-box">
          <div class="metric-val">{result['words']}</div>
          <div class="metric-lbl">총 단어(어절) 수</div>
        </div>
        <div class="metric-box">
          <div class="metric-val">{result['wpm']}</div>
          <div class="metric-lbl">말 속도 (WPM)</div>
        </div>
        <div class="metric-box">
          <div class="metric-val">{len(result['silences'])}</div>
          <div class="metric-lbl">침묵 구간 횟수</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 음성 인식 텍스트 ──
    st.markdown('<div class="piq-card"><div class="piq-section-label">음성 인식 텍스트</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="transcript">{result["text"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 침묵 구간 타임라인 (Plotly) ──
    st.markdown('<div class="piq-card"><div class="piq-section-label">침묵 구간 타임라인</div>', unsafe_allow_html=True)

    silences = result['silences']
    fig = go.Figure()

    # 전체 발표 구간 (파란색 배경 바)
    fig.add_trace(go.Bar(
        x=[dur], y=['발표'],
        orientation='h',
        marker_color='rgba(0,152,212,0.2)',
        showlegend=False, hoverinfo='skip',
    ))

    # 침묵 구간 (빨간 마커) — for 반복문
    for i, seg in enumerate(silences):
        fig.add_trace(go.Bar(
            x=[seg['duration']],
            y=['발표'],
            base=[seg['start']],
            orientation='h',
            marker_color='rgba(255,107,107,0.7)',
            name=f"침묵 #{i+1}",
            hovertemplate=f"침묵 #{i+1}<br>{seg['start']}s ~ {seg['end']}s<br>({seg['duration']}초)<extra></extra>",
        ))

    fig.update_layout(
        barmode='overlay',
        height=120,
        margin=dict(l=0, r=0, t=10, b=10),
        plot_bgcolor='rgba(22,32,48,1)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            range=[0, dur],
            tickfont=dict(color='#5a7a99', size=11),
            gridcolor='#1e3048',
            showline=False,
        ),
        yaxis=dict(visible=False),
        font=dict(color='#dce8f5'),
        legend=dict(
            orientation='h', y=-0.3,
            font=dict(color='#8ab4cc', size=11),
            bgcolor='rgba(0,0,0,0)',
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # 침묵 구간 목록 — for 반복문
    if silences:
        rows = ""
        for i, seg in enumerate(silences):
            rows += f"**#{i+1}** &nbsp; `{seg['start']}s ~ {seg['end']}s` &nbsp; ({seg['duration']}초)\n\n"
        st.markdown(rows)
    else:
        st.markdown('<span style="color:#5a7a99;font-size:.85rem">감지된 침묵 구간 없음</span>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── 피드백 ──
    st.markdown('<div class="piq-card"><div class="piq-section-label">개선 피드백</div>', unsafe_allow_html=True)
    fb_html = ""
    for fb in result['feedbacks']:   # for 반복문: 피드백 리스트 순회
        fb_html += f"""
        <div class="fb {fb['type']}">
          <div class="fb-cat">{fb['cat']}</div>
          <div class="fb-msg">{fb['msg']}</div>
        </div>"""
    st.markdown(fb_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 실행 결과 예시 (항상 표시) ──
with st.expander("📋 실행 결과 예시 보기"):
    st.markdown("""
<div class="example">
════════════════════════════════════════════<br>
&nbsp;&nbsp;🎙️ &nbsp;발표 분석 결과<br>
════════════════════════════════════════════<br>
📌 총 발표 시간 &nbsp;&nbsp;&nbsp;: <span style="color:#ffd166">87.4초</span><br>
📌 총 단어 수 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#ffd166">213 어절</span><br>
📌 말 속도 (WPM) : <span style="color:#ffd166">146.3 WPM</span><br>
📌 침묵 횟수 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#ffd166">2회</span><br>
────────────────────────────────────────────<br>
🗣 &nbsp;인식된 텍스트 :<br>
&nbsp;&nbsp;<span style="color:#ffd166">안녕하세요. 오늘 발표할 주제는 인공지능의 활용 방안입니다…</span><br>
────────────────────────────────────────────<br>
⏸ &nbsp;침묵 구간 :<br>
&nbsp;&nbsp;<span style="color:#ffd166">#1 : 12.3s ~ 14.1s (1.8초)</span><br>
&nbsp;&nbsp;<span style="color:#ffd166">#2 : 45.0s ~ 46.5s (1.5초)</span><br>
────────────────────────────────────────────<br>
💬 피드백 :<br>
&nbsp;&nbsp;<span style="color:#3ddc84">✅ [말 속도] 146.3 WPM — 적절한 발표 속도입니다!</span><br>
&nbsp;&nbsp;<span style="color:#3ddc84">✅ [발표 흐름] 침묵 구간이 적어 흐름이 매끄럽습니다!</span><br>
&nbsp;&nbsp;<span style="color:#3ddc84">✅ [발표 시간] 1분 27초 — 적절한 범위입니다!</span><br>
════════════════════════════════════════════
</div>
""", unsafe_allow_html=True)
