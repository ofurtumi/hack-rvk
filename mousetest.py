from time import sleep, time
import mouse, time

i = 0;
while True:
    mouse.move(100,0, absolute=True, duration=0.1)
    mouse.move(0,100, absolute=True, duration=0.1)
    mouse.move(-100,0, absolute=True, duration=0.1)
    mouse.move(0,-100, absolute=True, duration=0.1)
    time.sleep(0.2)

# import pyautogui

# pyautogui.moveRel(100, 100)