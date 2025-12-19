import os

import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS

if __name__ == '__main__':
    os.environ["CHATTERBOX_MODEL_REPO"] = "estevaofreitas/chatterbox-tts-ptbr"
    # Multilingual examples
    model = ChatterboxTTS.from_pretrained(device="cuda")

    # multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda",
    #                                                                model_name="estevaofreitas/chatterbox-tts-ptbr")

    french_text = "que o verdadeiro herdeiro Sheng Da"
    wav_french = model.generate(french_text)
    ta.save("test-french.wav", wav_french, model.sr)

    chinese_text = "你好，今天天气真不错，希望你有一个愉快的周末。"
    # wav_chinese = multilingual_model.generate(chinese_text, language_id="zh")
    # ta.save("test-chinese.wav", wav_chinese, multilingual_model.sr)
    # # If you want to synthesize with a different voice, specify the audio prompt
    # AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
    # wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
    # ta.save("test-2.wav", wav, model.sr)
