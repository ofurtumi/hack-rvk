from cmath import sqrt
import cv2, time, mediapipe as mp
from pyautogui import click
import mouse

mp_drawing = mp.solutions.drawing_utils         # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles # type: ignore
mp_hands = mp.solutions.hands                   # type: ignore

cam = cv2.VideoCapture("/dev/video2") # ? usb cam virkar ekki stundum??
# cam = cv2.VideoCapture(0) # ? built in cam virkar alltaf

item_pos = [100, 100]
item_size = 50

canvas=[]


def isFolded(coords_top, coords_bot):
    x = coords_top.x - coords_bot.x 
    y = coords_top.y - coords_bot.y

    x_pow2 = x * x
    y_pow2 = y * y

    sum = x_pow2 + y_pow2
    square = sqrt(sum)
    absolute = abs(square)

    folded = absolute < 0.03

    return {"folded": folded, "length": absolute}

# ? hendur skilgreindar í gegnum mediapipe safnið
hands_solution = mp.solutions.hands #type: ignore
hands = hands_solution.Hands(
    static_image_mode = False,
    # min_tracking_confidence = 0.75,   # * min tracking conf, default: 0.5
    # min_detection_confidence = 0.65,  # * min detect conf, default: 0.5
    max_num_hands = 1                 # * max hendur, default: 2
)
draw_solution = mp.solutions.drawing_utils #type: ignore

lastClick = 0

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
        hLen = result.multi_hand_world_landmarks[0].landmark
        hPos = result.multi_hand_landmarks[0].landmark
        grab = True;

        # ! pyautogui.moveTo((1-hPos[9].x)*1920, hPos[9].y*1080)
        mouse.move((1-hPos[9].x)*2*1920, hPos[9].y*2*1080)
        # print(hPos[8])

        for i in range(0,13,4):
            isit = isFolded(hLen[7+i], hLen[5+i])
            # print(f"{i/4}: {isit}")
            grab = grab & isit["folded"]

        print(lastClick)
        print(time.time())
        if grab and lastClick + 0.5 < time.time():
            # print(grab)
            click((1-hPos[9].x)*2*1920, hPos[9].y*2*1080)
            lastClick = time.time()
            print(lastClick)
            # mouse.press()
            # if item_pos[0] - 0.5*item_size <= hPos[9].x*img.shape[1] <= item_pos[0] + 0.5 * item_size and item_pos[1] - 0.5*item_size <= hPos[9].y*img.shape[0] <= item_pos[1] + 0.5 * item_size:
            #     print("get gripið")
                # item_pos = [int(hPos[9].x*img.shape[1]), int(hPos[9].y*img.shape[0])]
            # canvas.append(hPos[9]) # ? til að teikna 1

        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    cv2.circle(img, (item_pos[0], item_pos[1]), item_size, (0, 0, 255), cv2.FILLED)
            
    # ? til að teikna 2
    if canvas:
        for coords in canvas:
            # print(coords.x)
            h, w, c, = img.shape
            cx, cy = int(coords.x*w), int(coords.y*h)
            cv2.circle(img, (cx,cy), 10, (255,0,0), cv2.FILLED)

    # ? flippar kvikindinu
    cv2.imshow('Hendur - TEST', cv2.flip(img, 1))
    # cv2.imshow('Hendur - TEST', img)


    # ? ef fær input 'esc': break
    if cv2.waitKey(5) & 0xFF == 27:
        break

cam.release()