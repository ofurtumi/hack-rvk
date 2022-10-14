from cmath import sqrt
from email.mime import image
import cv2, time, mediapipe as mp

mp_drawing = mp.solutions.drawing_utils         # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles # type: ignore
mp_hands = mp.solutions.hands                   # type: ignore

# cam = cv2.VideoCapture("/dev/video3") # ? usb cam virkar ekki stundum??
cam = cv2.VideoCapture(0) # ? built in cam virkar alltaf

canvas=[]


def isFolded(coords):
    x_pow2 = coords.x  * coords.x
    y_pow2 = coords.y  * coords.y

    sum = x_pow2 + y_pow2
    square = sqrt(sum)
    absolute = abs(square)

    return absolute

# ? hendur skilgreindar í gegnum mediapipe safnið
hands_solution = mp.solutions.hands #type: ignore
hands = hands_solution.Hands(
    static_image_mode = False,
    # min_tracking_confidence = 0.75,   # * min tracking conf, default: 0.5
    # min_detection_confidence = 0.65,  # * min detect conf, default: 0.5
    max_num_hands = 1                 # * max hendur, default: 2
)
draw_solution = mp.solutions.drawing_utils #type: ignore

while cam.isOpened():
    isOn, img = cam.read()
    if not isOn:
        print("Ekkert inntak, sleppi ramma!")
        continue

    # ! tékk á höndum gerð áður en skrifað inn
    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # ? setja sem grátt
    result = hands.process(img)               # ? athuga hvort séu hendur

    # ! teikna UI inn á mynd, taka út fyrir bara gögn
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if result.multi_hand_landmarks:
        hLen = result.multi_hand_world_landmarks[0].landmark[12]
        # print(result.multi_handedness)
        # print(hLen)
        # print(getLen(hLen))
        # if (abs(result.multi_hand_world_landmarks[0].landmark[8].x) > 0.025):
        # canvas.append(result.multi_hand_landmarks[0].landmark[8])
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
    if canvas:
        # print(canvas[0].x*1920)
        for coords in canvas:
            h, w, c, = img.shape
            cx, cy = int(coords.x*w), int(coords.y*h)
            cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

    # ? flippar kvikindinu
    cv2.imshow('Hendur - TEST', cv2.flip(img, 1))
    # cv2.imshow('Hendur - TEST', img)


    # ? ef fær input 'esc': break
    if cv2.waitKey(5) & 0xFF == 27:
        break

cam.release()