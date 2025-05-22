from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from matcher import QuranMatcher
import os
import tempfile
from audio_extract import extract_audio

def get_router() -> Router:

    router = Router()

    @router.message(Command("start"))
    async def start_command(message: Message):
        welcome_text = (
            "🌙 *Assalamu alaikum!*\n\n"
            "Welcome to the *Qur'an Verse Finder Bot*. I can help you identify Qur'anic verses from recitations.\n\n"
            "*How to use:*\n"
            "📱 Send me a *voice recording* of the recitation\n"
            "🎵 Upload an *audio file* containing Qur'anic verses\n"
            "🎞️ Share a *video* of someone reciting the Qur'an\n\n"
            "I'll analyze the audio, transcribe it, and find the matching verses from the Holy Qur'an for you.\n\n"
            "May Allah bless your journey with the Qur'an. 🤲"
        )
        await message.answer(welcome_text, parse_mode="Markdown")

    @router.message(lambda msg: msg.voice or msg.audio or 
                   (msg.document and msg.document.mime_type and 
                    (msg.document.mime_type.startswith("audio/") or 
                     msg.document.mime_type.startswith("video/"))) or
                   (msg.video))
    async def find_verses(message: Message):
        msg = await message.answer("🔄 Started processing...")
        
        # Determine the file type and get the file
        file = None
        is_video = False
        
        if message.voice:
            file = message.voice
        elif message.audio:
            file = message.audio
        elif message.video:
            file = message.video
            is_video = True
        elif message.document:
            file = message.document
            is_video = file.mime_type and file.mime_type.startswith("video/")

        if not file:
            await msg.edit_text("❌ Unsupported file format.")
            return

        try:
            # Get file info and path
            file_info = await message.bot.get_file(file.file_id)
            file_path = file_info.file_path
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_tmp:
                audio_path = audio_tmp.name
            
            # For video files, we need to extract audio first
            if is_video:
                await msg.edit_text("🎬 Extracting audio from video...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_tmp:
                    await message.bot.download_file(file_path, destination=video_tmp)
                    video_path = video_tmp.name
                
                # Extract audio from video
                extract_audio(video_path, audio_path, overwrite=True)
                
                # Clean up the video file
                os.remove(video_path)
            else:
                # Direct download for audio files
                await message.bot.download_file(file_path, destination=audio_path)

            matcher = QuranMatcher()

            await msg.edit_text("🎧 Transcribing audio...")
            transcript = matcher.transcribe(audio_path)

            # Clean up the audio file
            os.remove(audio_path)

            await msg.edit_text("🔍 Searching the verses...")
            result = matcher.match(transcript)

            if result:
                await msg.edit_text(f"✅ Process completed successfully.")

                result_text = "\n".join([f"Surah {v['surah']} Ayah {v['verse']}:\n{v['text']}" for v in result])

                await message.reply(result_text)
            else:
                await msg.edit_text("❌ No verses found.")

        except Exception:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error processing media: {error_trace}")
            await msg.edit_text(f"❌ Error processing the file.")

    return router