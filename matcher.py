import re
import torch
import torchaudio
from rapidfuzz import fuzz
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class QuranMatcher:
    def __init__(self, quran_file="quran.txt", model_name="tarteel-ai/whisper-base-ar-quran"):
        self.quran_verses = []
        self.quran_dict = {}
        self.verse_lookup = {}

        self.load_quran(quran_file)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    # ---------- Text Normalization ----------
    def remove_diacritics_and_symbols(self, text):
        harakat = re.compile(r'[\u064B-\u0652]')
        quran_symbols = re.compile(r'[۝۞۩ۭۨۦ۪ۧۡۢ۠ۤ۟۫۬ۥ۫ۘۚۛۙۗۖ]')
        text = harakat.sub('', text)
        text = quran_symbols.sub('', text)
        return text.replace('ـ', '')

    def normalize_text(self, text):
        text = self.remove_diacritics_and_symbols(text)
        text = re.sub(r'[^\u0621-\u063A\u0641-\u064A\s]', '', text)
        return re.sub(r'\s+', ' ', text.strip())

    # ---------- Quran Loader ----------
    def load_quran(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|", 2)
                if len(parts) != 3:
                    continue
                s, v, txt = int(parts[0]), int(parts[1]), parts[2]

                if v == 1:
                    if s == 1:
                        norm = self.normalize_text(txt)
                    elif s != 9 and txt.startswith("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"):
                        trimmed = txt[len("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"):].strip()
                        norm = self.normalize_text(trimmed)
                    else:
                        norm = self.normalize_text(txt)
                else:
                    norm = self.normalize_text(txt)

                norm_words = norm.split()
                self.quran_verses.append((s, v, txt.strip(), norm, norm_words, len(norm_words)))
                self.quran_dict[(s, v)] = (s, v, txt.strip(), norm)
                self.verse_lookup[(s, v)] = (txt.strip(), norm, norm_words, len(norm_words))

    # ---------- Transcription ----------
    def transcribe(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        target_sample_rate = 16000

        if sample_rate != target_sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        chunk_size = target_sample_rate * 30
        num_chunks = (waveform.shape[1] + chunk_size - 1) // chunk_size
        transcriptions = []

        for i in range(num_chunks):
            chunk = waveform[:, i * chunk_size:(i + 1) * chunk_size]
            input_features = self.processor(chunk.squeeze().numpy(), sampling_rate=target_sample_rate, return_tensors="pt").input_features.to(self.model.device)

            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)

            text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcriptions.append(text)

        full = " ".join(transcriptions)
        return full

    # ---------- Fuzzy Matcher ----------
    def check_best_match_window(self, search_pattern, words, threshold=75):
        word_count = len(search_pattern.split())
        best_score = 0
        best_text = ""
        best_position = -1

        min_w = max(1, word_count - max(1, word_count // 3))
        max_w = min(len(words), word_count + max(2, word_count // 2))

        for w in range(min_w, max_w + 1):
            for i in range(len(words) - w + 1):
                chunk = " ".join(words[i:i + w])
                score = fuzz.token_sort_ratio(search_pattern, chunk)
                if score > best_score:
                    best_score = score
                    best_text = chunk
                    best_position = i

        return best_score >= threshold, best_text, best_score, best_position

    # ---------- Match Quran ----------
    def match(self, transcription, min_sequence_len=2):
        transcribed_norm = self.normalize_text(transcription)
        transcribed_words = transcribed_norm.split()

        matches = []
        for s, v, orig_text, norm, norm_words, word_len in self.quran_verses:
            found, match_text, score, pos = self.check_best_match_window(norm, transcribed_words)
            if found:
                matches.append({
                    "surah": s,
                    "verse": v,
                    "text": orig_text,
                    "norm": norm,
                    "norm_words": norm_words,
                    "word_len": word_len,
                    "score": score,
                    "position": pos
                })

        matches.sort(key=lambda x: x['position'])

        best_sequence = []
        for i in range(len(matches)):
            m = matches[i]
            seq = [m]
            cur_s, cur_v = m['surah'], m['verse']
            pos = m['position'] + m['word_len']

            while True:
                next_key = (cur_s, cur_v + 1)
                if next_key not in self.verse_lookup:
                    break

                next_text, next_norm, next_words, next_len = self.verse_lookup[next_key]
                found, _, score, rel_pos = self.check_best_match_window(next_norm, transcribed_words[pos:])
                if found:
                    abs_pos = pos + rel_pos
                    seq.append({
                        "surah": next_key[0],
                        "verse": next_key[1],
                        "text": next_text,
                        "norm": next_norm,
                        "norm_words": next_words,
                        "word_len": next_len,
                        "score": score,
                        "position": abs_pos
                    })
                    cur_s, cur_v = next_key
                    pos = abs_pos + next_len
                else:
                    break

            if len(seq) > len(best_sequence):
                best_sequence = seq
            elif len(seq) == len(best_sequence):
                new_avg = sum(x['score'] for x in seq) / len(seq)
                old_avg = sum(x['score'] for x in best_sequence) / len(best_sequence)
                if new_avg > old_avg:
                    best_sequence = seq

        if best_sequence and len(best_sequence) >= min_sequence_len:
            return best_sequence
        return []


# ---------- Example usage ----------
if __name__ == "__main__":
    matcher = QuranMatcher()

    print("\n🎧 Transcribing audio...")
    transcript = matcher.transcribe("audio.mp3")
    print("\n🧾 Transcribed text:")
    print(transcript)

    print("\n🔍 Finding best Quran matches...")
    result = matcher.match(transcript)

    if result:
        print(f"\n✅ Found {len(result)} matching sequential verses:")
        for v in result:
            print(f"Surah {v['surah']} Ayah {v['verse']} ({v['score']}%): {v['text']}")
    else:
        print("❌ No valid sequence found.")
