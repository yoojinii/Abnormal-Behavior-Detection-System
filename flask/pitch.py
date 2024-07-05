from pydub import AudioSegment

# 음악 파일 불러오기
audio = AudioSegment.from_file('oneperson_ko.mp3')

# 피치 조절 (예: 1.5배로 높임)
pitch_adjusted = audio._spawn(audio.raw_data, overrides={
    "frame_rate": int(audio.frame_rate * 1.5)
})

# 조절된 파일 저장
pitch_adjusted.export('oneperson_adjusted_music.mp3', format="mp3")