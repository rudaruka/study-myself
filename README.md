# 🎙️ PresentIQ — 발표 능력 분석 도우미

> 음성을 업로드하거나 마이크로 녹음하면 **말 속도 · 침묵 구간 · 발표 흐름**을 자동 분석하는 Streamlit 웹 앱입니다.

---

## 📁 파일 구조

```
presentation_analyzer/
├── app.py              # Streamlit 메인 앱 (분석 로직 + UI 전체)
├── requirements.txt    # 필요 라이브러리
├── .gitignore
└── README.md
```

---

## ⚙️ 로컬 실행

### 1단계 — 저장소 클론
```bash
git clone https://github.com/사용자명/저장소명.git
cd 저장소명
```

### 2단계 — ffmpeg 설치 (pydub 필수 의존성)

| OS | 명령어 |
|----|--------|
| macOS | `brew install ffmpeg` |
| Ubuntu | `sudo apt install ffmpeg` |
| Windows | https://ffmpeg.org/download.html 다운로드 후 PATH 등록 |

### 3단계 — 패키지 설치
```bash
pip install -r requirements.txt
```

### 4단계 — 실행
```bash
streamlit run app.py
```
→ 브라우저에서 **http://localhost:8501** 자동 접속

---

## 🚀 GitHub + Streamlit Cloud 배포

### 1단계 — GitHub 저장소 생성 & 푸시
```bash
git init
git add .
git commit -m "feat: 발표 분석 앱 초기 커밋"
git remote add origin https://github.com/사용자명/저장소명.git
git push -u origin main
```

### 2단계 — Streamlit Cloud 배포

1. **https://share.streamlit.io** 접속 후 GitHub 계정으로 로그인
2. **"New app"** 버튼 클릭
3. 아래 정보 입력:

| 항목 | 값 |
|------|-----|
| Repository | `사용자명/저장소명` |
| Branch | `main` |
| Main file path | `app.py` |

4. **"Deploy!"** 클릭 → 자동 빌드 & 배포 완료
5. `https://사용자명-저장소명-app-xxxx.streamlit.app` 형태의 URL로 공개 접근 가능

> ⚠️ **Streamlit Cloud의 ffmpeg 지원**
> Streamlit Cloud는 ffmpeg가 기본 설치되어 있어 별도 설정 없이 pydub가 작동합니다.

---

## 🔍 분석 기준

| 지표 | 기준 |
|------|------|
| 말 속도 | **80~180 WPM** → 적절 / 미달·초과 시 경고 |
| 침묵 구간 | **3회 이하** → 양호 / 5회 초과 → 경고 |
| 발표 시간 | **30초 ~ 10분** → 적절 범위 |

---

## 📊 실행 결과 예시

```
════════════════════════════════════════════
  🎙️  발표 분석 결과
════════════════════════════════════════════
📌 총 발표 시간   : 87.4초
📌 총 단어 수     : 213 어절
📌 말 속도 (WPM)  : 146.3 WPM
📌 침묵 구간 횟수 : 2회
────────────────────────────────────────────
🗣  인식된 텍스트 :
  안녕하세요. 오늘 발표할 주제는 인공지능의 활용 방안입니다...
────────────────────────────────────────────
⏸  침묵 구간 목록 :
  #1 : 12.3s ~ 14.1s  (1.8초)
  #2 : 45.0s ~ 46.5s  (1.5초)
────────────────────────────────────────────
💬 피드백 :
  ✅ [말 속도] 146.3 WPM — 적절한 발표 속도입니다!
  ✅ [발표 흐름] 침묵 구간이 적어 흐름이 매끄럽습니다!
  ✅ [발표 시간] 1분 27초 — 적절한 범위입니다!
════════════════════════════════════════════
```

---

## ⚠️ 주의사항

- 음성 인식은 **인터넷 연결** 필요 (Google Web Speech API)
- 지원 오디오 형식: `mp3`, `wav`, `m4a`, `ogg`, `webm`, `flac`
- 마이크 녹음 시 브라우저 마이크 권한 허용 필요
