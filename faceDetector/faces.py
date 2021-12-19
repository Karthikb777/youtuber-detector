import numpy as np
import cv2
import pickle
from speechRecog import takeCommand, wishMe, speak


class youtuber_detector:
    youtuber_ids = {
        "linus tech tips": 1,
        "austin evans": 2,
        "ijustine": 3,
        "dave lee": 4,
        "mkbhd": 5,
        "bhuvan bam": 6,
        "ashish chanchalani": 7
    }

    people_in_vid = set()

    def init_voice(self):
        # wishMe()
        while True:
            # speak("Tell me how can I help you now?")
            statement = takeCommand().lower()

            if statement == 0:
                continue

            if 'open camera' in statement:
                self.recognize_faces()
                break

    def multiple_faces_warning(self):  # returns the youtuber's name.
        speak("Multiple faces detected, whose video do you want to play?")
        print("Multiple faces detected, whose video do you want to play?\n")
        for person in self.people_in_vid:
            print(f"{self.youtuber_ids[person]} - {person}")

        print("\ntell the id of the youtuber.......")
        statement = takeCommand().lower()
        print(f"user said: {statement}")

        yt_id_reversed = {v:k for k, v in self.youtuber_ids.items()}
        print(f"playing {yt_id_reversed[int(statement)]} video....")
        return yt_id_reversed[int(statement)]

    def recognize_faces(self):

        # trained models directory
        MODEL_DIR = "..\\resources\\models\\"

        # saved pickle directory
        PICKLE_DIR = "..\\resources\\pickles\\"

        # haar cascades classifier
        face_cascade = cv2.CascadeClassifier('../resources/cascades/data/haarcascade_frontalface_alt2.xml')

        # face recognizer
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        # trained data
        face_recognizer.read(MODEL_DIR + "face_recognizer.yml")

        # opening the labels pickle
        labels = dict()
        with open(PICKLE_DIR + "labels.pickle", "rb") as file:
            labels = pickle.load(file)
        labels = {v: k for k, v in labels.items()}

        # getting the video sources
        cap = cv2.VideoCapture(0)
        cap2 = None

        video_to_show = " "
        showing_vid = False
        multiple_people = False

        while cap.isOpened():
            ret, frame = cap.read()

            frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)

            """
                this will give us the face
                x will be the x coordinate, y will be the y coordinate, 
                w will be the width of the face and h will be the height of the face
                so if we take a slice of y to y+h and x to x+h, we can get the face that is detected 
                in the frame.
            """

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_clr = frame[y:y + h, x:x + w]

                # resize the roi_gray for better detection
                roi_gray = cv2.resize(roi_gray, (550, 550))

                # recognizing the faces
                id_, confidence = face_recognizer.predict(roi_gray)
                if confidence >= 45:
                    name = labels[id_]
                    video_to_show = name
                    self.people_in_vid.add(name)

                    # detecting multiple people in the video
                    if len(self.people_in_vid) > 1 and not multiple_people:
                        video_to_show = self.multiple_faces_warning()

                        # THE OLD COMMAND LINE OPTION
                        # print("multiple people detected")
                        # print("whose video do you wanna play?")
                        # for n in people_in_vid:
                        #     print(n)
                        # video_to_show = input("enter the name: ")
                        # print(video_to_show)

                        showing_vid = False
                        multiple_people = True
                        cv2.waitKey(500)

                    # adding the second video stream to show
                    if not showing_vid:
                        cap2 = cv2.VideoCapture(f"../resources/videos/{video_to_show}.mp4")
                        showing_vid = True

                    # showing the second video
                    r2, frame2 = cap2.read()
                    frame2 = cv2.resize(frame2, (w + 100, h))

                    try:
                        frame[y:y + h, x:x + w + 100] = frame2
                    except ValueError as err:

                        # THIS IS NOT REQUIRED
                        # out_of_bounds = str(err).split()[-1]
                        # out_of_bounds = out_of_bounds.rstrip(')').lstrip('(').split(',')
                        # out_of_bounds = [int(i) for i in out_of_bounds]
                        # print(f" going out of bounds: {out_of_bounds}")

                        frame[y:y+h, x-150: x+w+100-150] = frame2

                    if cv2.waitKey(3) == ord('q'):
                        break

                    # cv2.putText(frame, video_to_show, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                #     specifying the color to draw a rectangle around the face
                # TODO: DONE instead of drawing a rectangle on the face, replace the face with the youtube video
                # color = (255, 0, 0)  # (blue, green, red) 0-255
                # stroke = 2
                # end_cord_x = x + w
                # end_cord_y = y + h
                # cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color=color, thickness=stroke)
            # cv2.copyMakeBorder(frame, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 200, 200])

            cv2.imshow("frame", frame)

            if cv2.waitKey(10) == ord('q'):
                break

        cap.release()
        cap2.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    debug = False

    detector = youtuber_detector()

    if debug:
        detector.recognize_faces()
    else:
        detector.init_voice()
