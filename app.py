# ============================================================
#  발표 능력 분석 웹 도우미 프로그램
#  -- Flask 기반 백엔드 --
#
#  필요 라이브러리:
#    pip install flask speechrecognition pydub librosa
#    (ffmpeg 설치 필요: https://ffmpeg.org/)
# ============================================================

from flask import Flask, render_template, request, jsonify
import speech_recognition as sr   # 음성 → 텍스트 변환
from pydub import AudioSegment     # 오디오 파일 처리
from pydub.silence import detect_silence  # 침묵 구간 감지
import librosa                     # 오디오 길이·특성 분석
import os
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 최대 50MB 업로드 허용


# ─────────────────────────────────────────────
# [함수 1] 오디오 파일 → WAV 형식으로 변환
# ─────────────────────────────────────────────
def convert_to_wav(filepath: str) -> str:
    """
    다양한 오디오 형식(mp3, m4a, webm 등)을 WAV로 변환한다.
    SpeechRecognition 라이브러리는 WAV 파일을 기본으로 사용한다.
    """
    audio = AudioSegment.from_file(filepath)  # pydub로 파일 읽기
    wav_path = filepath.rsplit('.', 1)[0] + '_converted.wav'
    audio.export(wav_path, format='wav')       # WAV 포맷으로 저장
    return wav_path


# ─────────────────────────────────────────────
# [함수 2] 음성 파일 → 텍스트 변환 (STT)
# ─────────────────────────────────────────────
def transcribe_audio(wav_path: str) -> str:
    """
    SpeechRecognition + Google Web Speech API를 이용해
    음성을 한국어 텍스트로 변환한다.
    인터넷 연결이 필요하다.
    """
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(wav_path) as source:
            # 주변 소음 수준에 맞게 인식기 감도 자동 조정
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)  # 전체 파일 읽기

        # Google Web Speech API로 한국어 인식
        text = recognizer.recognize_google(audio_data, language='ko-KR')
        return text

    except sr.UnknownValueError:
        # 음성을 알아들을 수 없는 경우
        return ""
    except sr.RequestError as e:
        # API 연결 실패
        raise ConnectionError(f"음성 인식 API 오류: {e}")


# ─────────────────────────────────────────────
# [함수 3] 단어 수 계산
# ─────────────────────────────────────────────
def count_words(text: str) -> int:
    """
    변환된 텍스트에서 단어(어절) 수를 계산한다.
    한국어는 띄어쓰기 단위로 어절을 분리한다.
    """
    if not text:
        return 0

    # 리스트 활용: 텍스트를 공백으로 분리해 단어 리스트 생성
    word_list = text.split()
    return len(word_list)


# ─────────────────────────────────────────────
# [함수 4] 말 속도 계산 (WPM: Words Per Minute)
# ─────────────────────────────────────────────
def calculate_speech_speed(word_count: int, duration_seconds: float) -> float:
    """
    분당 단어 수(WPM)를 계산한다.
    WPM = 단어 수 / 발표 시간(분)
    """
    if duration_seconds <= 0:
        return 0.0

    duration_minutes = duration_seconds / 60  # 초 → 분 변환
    wpm = word_count / duration_minutes
    return round(wpm, 1)


# ─────────────────────────────────────────────
# [함수 5] 침묵 구간 감지
# ─────────────────────────────────────────────
def detect_silence_segments(wav_path: str,
                             min_silence_len: int = 1000,
                             silence_thresh: int = -40) -> list:
    """
    1초(1000ms) 이상 조용한 구간을 감지한다.
    - min_silence_len : 최소 침묵 길이 (ms 단위)
    - silence_thresh  : 침묵으로 판단할 음량 기준 (dBFS)

    반환: 침묵 구간 정보가 담긴 리스트
    """
    audio = AudioSegment.from_wav(wav_path)

    # pydub의 detect_silence 함수로 침묵 구간 탐지
    silent_ranges = detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )

    # 리스트 활용: 각 침묵 구간을 딕셔너리로 변환해 리스트에 저장
    silence_list = []
    for start_ms, end_ms in silent_ranges:
        silence_list.append({
            'start':    round(start_ms / 1000, 1),          # ms → 초
            'end':      round(end_ms   / 1000, 1),
            'duration': round((end_ms - start_ms) / 1000, 1)
        })

    return silence_list


# ─────────────────────────────────────────────
# [함수 6] 오디오 전체 길이 계산
# ─────────────────────────────────────────────
def get_audio_duration(wav_path: str) -> float:
    """
    librosa를 이용해 오디오 파일의 총 길이(초)를 계산한다.
    """
    y, sr_rate = librosa.load(wav_path, sr=None)  # 원본 샘플레이트 유지
    duration = librosa.get_duration(y=y, sr=sr_rate)
    return round(duration, 2)


# ─────────────────────────────────────────────
# [함수 7] 피드백 메시지 생성
# ─────────────────────────────────────────────
def generate_feedback(wpm: float, silence_count: int, duration: float) -> list:
    """
    분석 결과를 바탕으로 맞춤형 피드백 메시지를 생성한다.
    - if 조건문으로 각 지표를 판단
    - 결과를 리스트로 반환
    """
    feedbacks = []  # 리스트: 피드백을 순서대로 저장

    # ── 말 속도 피드백 (한국어 발표 기준: 80~180 WPM 적정) ──
    if wpm > 180:
        feedbacks.append({
            'type':     'warning',
            'category': '🚀 말 속도',
            'message':  (f'말 속도({wpm} WPM)가 너무 빠릅니다. '
                         '청중이 내용을 따라가기 어려울 수 있습니다. '
                         '의식적으로 천천히, 또렷하게 말하는 연습을 해보세요.')
        })
    elif wpm < 80 and wpm > 0:
        feedbacks.append({
            'type':     'warning',
            'category': '🐢 말 속도',
            'message':  (f'말 속도({wpm} WPM)가 너무 느립니다. '
                         '청중의 집중력이 흐트러질 수 있습니다. '
                         '자신감 있게 또렷한 발음으로 말해보세요.')
        })
    elif wpm == 0:
        feedbacks.append({
            'type':     'warning',
            'category': '🎙️ 음성 인식',
            'message':  '음성이 인식되지 않았습니다. 마이크 상태나 발음을 확인해보세요.'
        })
    else:
        feedbacks.append({
            'type':     'good',
            'category': '✅ 말 속도',
            'message':  (f'말 속도({wpm} WPM)가 적절합니다. '
                         '청중이 내용을 이해하기 좋은 속도입니다!')
        })

    # ── 침묵 구간 피드백 ──
    if silence_count > 5:
        feedbacks.append({
            'type':     'warning',
            'category': '⏸️ 침묵 구간',
            'message':  (f'1초 이상의 침묵이 {silence_count}회 감지되었습니다. '
                         '발표 내용을 충분히 암기하고, "음…", "어…" 습관을 줄이는 연습을 해보세요.')
        })
    elif silence_count > 2:
        feedbacks.append({
            'type':     'caution',
            'category': '⚠️ 침묵 구간',
            'message':  (f'침묵 구간이 {silence_count}회 감지되었습니다. '
                         '핵심 문장을 미리 암기하면 자연스러운 흐름을 만들 수 있습니다.')
        })
    else:
        feedbacks.append({
            'type':     'good',
            'category': '✅ 발표 흐름',
            'message':  '침묵 구간이 거의 없어 발표 흐름이 매끄럽습니다!'
        })

    # ── 발표 시간 피드백 ──
    if duration < 30:
        feedbacks.append({
            'type':     'caution',
            'category': '⏱️ 발표 시간',
            'message':  '발표 시간이 30초 미만으로 짧습니다. 내용을 더 풍부하게 구성해보세요.'
        })
    elif duration > 600:
        feedbacks.append({
            'type':     'caution',
            'category': '⏱️ 발표 시간',
            'message':  '발표 시간이 10분을 초과합니다. 핵심 내용 위주로 간결하게 구성하면 더 효과적입니다.'
        })
    else:
        mins = int(duration // 60)
        secs = int(duration % 60)
        feedbacks.append({
            'type':     'good',
            'category': '✅ 발표 시간',
            'message':  f'발표 시간({mins}분 {secs}초)이 적절한 범위입니다!'
        })

    return feedbacks


# ─────────────────────────────────────────────
# Flask 라우트 정의
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """메인 페이지 렌더링"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    음성 파일을 받아 전체 분석을 수행하고 JSON으로 결과 반환.
    프론트엔드에서 FormData로 파일을 전송한다.
    """
    # 파일 유효성 검사
    if 'audio' not in request.files:
        return jsonify({'error': '음성 파일이 없습니다.'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400

    # 업로드 파일을 임시 경로에 저장
    ext = os.path.splitext(audio_file.filename)[1] or '.wav'
    tmp_path = None
    wav_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        # 단계별 분석 수행 (반복문은 각 함수 내부에서 활용)
        wav_path  = convert_to_wav(tmp_path)          # 1. WAV 변환
        duration  = get_audio_duration(wav_path)       # 2. 총 길이
        text      = transcribe_audio(wav_path)         # 3. STT
        words     = count_words(text)                  # 4. 단어 수
        wpm       = calculate_speech_speed(words, duration)  # 5. WPM
        silences  = detect_silence_segments(wav_path)  # 6. 침묵 감지
        feedbacks = generate_feedback(wpm, len(silences), duration)  # 7. 피드백

        # 결과 딕셔너리 구성
        result = {
            'duration':      duration,
            'text':          text if text else '(음성 인식 실패 — 발음이나 마이크를 확인해주세요)',
            'word_count':    words,
            'wpm':           wpm,
            'silence_count': len(silences),
            'silences':      silences,
            'feedbacks':     feedbacks
        }
        return jsonify(result)

    except ConnectionError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'분석 중 오류 발생: {str(e)}'}), 500

    finally:
        # 임시 파일 반드시 삭제 (메모리 누수 방지)
        for path in [tmp_path, wav_path]:
            if path and os.path.exists(path):
                os.unlink(path)


# ─────────────────────────────────────────────
# 실행 진입점
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  🎙️  발표 분석 프로그램 서버 시작")
    print("  http://127.0.0.1:5000 에서 접속하세요")
    print("=" * 50)
    app.run(debug=True, port=5000)
