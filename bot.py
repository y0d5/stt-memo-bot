import os
import re
import logging
import tempfile
import asyncio
import requests
from pathlib import Path

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from openai import OpenAI
from pydub import AudioSegment

# ── 로깅 설정 ──────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── 환경변수 ────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

# ── 상수 ────────────────────────────────────────────────────
MAX_FILE_MB = 24          # Whisper API 25MB 제한보다 약간 낮게
CHUNK_MINUTES = 10        # 긴 파일 분할 단위
DEFAULT_LANG = "ko"       # 기본 언어


# ── 헬퍼: 오디오 파일 → 텍스트 ──────────────────────────────
def transcribe_file(path: str, language: str = DEFAULT_LANG) -> str:
    """단일 파일을 Whisper API로 변환 (타임스탬프 포함)"""
    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    lines = []
    current_texts = []
    current_start = None
    MERGE_SECONDS = 30  # 30초 단위로 묶기

    for seg in result.segments:
        start = int(seg.start)
        if current_start is None:
            current_start = start
        if start - current_start >= MERGE_SECONDS:
            mm, ss = divmod(current_start, 60)
            lines.append(f"[{mm:02d}:{ss:02d}] {''.join(current_texts).strip()}")
            current_texts = []
            current_start = start
        current_texts.append(seg.text)

    if current_texts:
        mm, ss = divmod(current_start, 60)
        lines.append(f"[{mm:02d}:{ss:02d}] {''.join(current_texts).strip()}")

    return "\n".join(lines)


def split_and_transcribe(path: str, language: str = DEFAULT_LANG) -> str:
    """25MB 초과 파일은 10분 단위로 분할 후 순차 변환"""
    audio = AudioSegment.from_file(path)
    chunk_ms = CHUNK_MINUTES * 60 * 1000
    texts = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, start in enumerate(range(0, len(audio), chunk_ms)):
            chunk = audio[start : start + chunk_ms]
            chunk_path = os.path.join(tmpdir, f"chunk_{idx:03d}.ogg")
            chunk.export(chunk_path, format="ogg")
            logger.info(f"청크 {idx+1} 변환 중…")
            texts.append(transcribe_file(chunk_path, language))

    return "\n".join(texts)


async def process_audio(update: Update, file_id: str, language: str = DEFAULT_LANG):
    """공통 오디오 처리 파이프라인"""
    msg = update.effective_message
    status = await msg.reply_text("🎙️ 파일 수신 중…")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 파일 다운로드
        tg_file = await update.get_bot().get_file(file_id)
        suffix = Path(tg_file.file_path).suffix or ".ogg"
        local_path = os.path.join(tmpdir, f"audio{suffix}")
        await tg_file.download_to_drive(local_path)

        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"다운로드 완료: {size_mb:.1f} MB")

        await status.edit_text("⚙️ 변환 중… (파일 크기에 따라 수십 초 소요)")

        if size_mb > MAX_FILE_MB:
            text = split_and_transcribe(local_path, language)
        else:
            text = transcribe_file(local_path, language)

    await status.delete()

    # 4096자 초과 시 분할 전송
    for i in range(0, len(text), 4000):
        await msg.reply_text(text[i : i + 4000])


# ── 커맨드 핸들러 ────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎙️ *STT 봇에 오신 걸 환영합니다!*\n\n"
        "음성 메시지 또는 오디오 파일을 전송하면 텍스트로 변환해 드립니다.\n\n"
        "📌 *지원 포맷:* m4a · mp3 · wav · ogg · flac · webm\n"
        "📌 *기본 언어:* 한국어 (자동 감지도 됩니다)\n\n"
        "━━━━━━━━━━━━━━━\n"
        "/lang en — 영어 모드\n"
        "/lang ko — 한국어 모드 (기본)\n"
        "/lang ja — 일본어 모드\n"
        "/lang auto — 자동 감지\n",
        parse_mode="Markdown",
    )


async def cmd_lang(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("사용법: /lang ko | en | ja | auto")
        return
    lang = context.args[0].lower()
    valid = {"ko", "en", "ja", "zh", "auto"}
    if lang not in valid:
        await update.message.reply_text(f"지원 언어: {', '.join(valid)}")
        return
    context.user_data["lang"] = lang if lang != "auto" else None
    label = {"ko": "한국어", "en": "영어", "ja": "일본어", "zh": "중국어", "auto": "자동 감지"}.get(lang, lang)
    await update.message.reply_text(f"✅ 언어 설정: {label}")


# ── 메시지 핸들러 ────────────────────────────────────────────
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """텔레그램 내 마이크 녹음 (voice)"""
    lang = context.user_data.get("lang", DEFAULT_LANG)
    await process_audio(update, update.message.voice.file_id, lang)


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """첨부 오디오 파일 (audio)"""
    lang = context.user_data.get("lang", DEFAULT_LANG)
    await process_audio(update, update.message.audio.file_id, lang)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """document 타입으로 전송된 오디오 파일"""
    doc = update.message.document
    audio_exts = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm", ".mp4"}
    suffix = Path(doc.file_name or "").suffix.lower()
    if suffix not in audio_exts:
        await update.message.reply_text("⚠️ 지원하지 않는 파일 형식입니다.\n지원 포맷: mp3 · m4a · wav · ogg · flac · webm")
        return
    lang = context.user_data.get("lang", DEFAULT_LANG)
    await process_audio(update, doc.file_id, lang)


# ── 구글 드라이브 링크 처리 ──────────────────────────────────
def extract_gdrive_id(url: str) -> str | None:
    """구글 드라이브 URL에서 파일 ID 추출"""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_gdrive_file(file_id: str, dest_path: str) -> bool:
    """구글 드라이브 파일 직접 다운로드"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(url, stream=True)

    # 대용량 파일 확인 토큰 처리
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
            response = session.get(url, stream=True)
            break

    if response.status_code != 200:
        return False

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    return True


async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """텍스트 메시지에서 구글 드라이브 링크 감지 후 처리"""
    text = update.message.text or ""
    if "drive.google.com" not in text:
        await update.message.reply_text("⚠️ 구글 드라이브 링크만 지원됩니다.")
        return

    file_id = extract_gdrive_id(text)
    if not file_id:
        await update.message.reply_text("⚠️ 링크에서 파일 ID를 찾을 수 없습니다.")
        return

    lang = context.user_data.get("lang", DEFAULT_LANG)
    msg = update.effective_message
    status = await msg.reply_text("🔗 구글 드라이브에서 다운로드 중…")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, "audio.m4a")
        success = download_gdrive_file(file_id, local_path)

        if not success:
            await status.edit_text("❌ 다운로드 실패. 파일 공유 설정을 '링크가 있는 모든 사용자'로 변경해 주세요.")
            return

        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        await status.edit_text(f"⚙️ 다운로드 완료 ({size_mb:.1f}MB), 변환 중…")

        if size_mb > MAX_FILE_MB:
            text_result = split_and_transcribe(local_path, lang)
        else:
            text_result = transcribe_file(local_path, lang)

    await status.delete()
    for i in range(0, len(text_result), 4000):
        await msg.reply_text(text_result[i : i + 4000])


# ── 메인 ────────────────────────────────────────────────────
def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("lang", cmd_lang))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_url))

    logger.info("STT 봇 시작")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
