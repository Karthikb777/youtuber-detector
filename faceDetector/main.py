from faces import recognize_faces
from speechRecog import takeCommand


def start():
    while True:
        statement = takeCommand().lower()
        if statement == 0:
            continue

        if 'open camera' in statement:
            recognize_faces()
            break


if __name__ == '__main__':
    start()
