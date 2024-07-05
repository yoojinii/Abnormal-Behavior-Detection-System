from flask import Flask, render_template
import pygame

application = Flask(__name__, static_folder='static')
def play_sound():
    pygame.mixer.init()
    pygame.mixer.music.load('oneperson_ko.mp3')
    pygame.mixer.music.play()

@application.route('/')
def index_1():
    play_sound()
    condition = True  # 조건을 여기에 맞게 설정해주세요

    if condition:
        alert_message = "이것은 경고 메시지입니다!"  # 경고 메시지 내용

        # 템플릿 렌더링 시에 경고 메시지를 전달합니다.
        return render_template('playsound.html', alert_message=alert_message)
    else:
        return render_template('playsound.htmll', alert_message=None)

if __name__ == '__main__':
    application.run(debug=True)