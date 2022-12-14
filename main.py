import cv2 
import os
from time import time, sleep
from windowcapture import WindowCapture
from vision import Vision
from threading import Thread
import ctypes, pyautogui


os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 17, 0, 0, 179, 255, 255, 0, 255, 105, 0

#WindowCapture.list_window_names()

wincap = WindowCapture("windowname")
vision_metin = Vision(None)

cascade_metin = cv2.CascadeClassifier('cascade/cascade.xml')

is_bot_in_action = False

MOUSE_LEFTDOWN = 0X0002
MOUSE_LEFTUP = 0X0004

def click():
    ctypes.windll.user32.mouse_event(MOUSE_LEFTDOWN)
    sleep(0.1)
    ctypes.windll.user32.mouse_event(MOUSE_LEFTUP)

def bot_action(rectangles):
    
    targets = vision_metin.get_click_points(rectangles)
    target = wincap.get_screen_position(targets[0])
    pyautogui.click(target[0], target[1])
    


   
    sleep(5)
    global is_bot_in_action
    is_bot_in_action = False

fps_start = 0
fps = 0
while (True):

    screenshot = wincap.get_screenshot()
    # object detection
    rectangles = cascade_metin.detectMultiScale(screenshot)

    # draw the detection results onto the original screenshot
    output_image = vision_metin.draw_rectangles(screenshot, rectangles)

    # display the processed image
    cv2.imshow('processed', output_image)

    if not is_bot_in_action:
        is_bot_in_action = True
        t = Thread(target=bot_action, args=(rectangles,))
        t.start()

    fps_end = time()
    time_diff = fps_end - fps_start
    fps = 1/(time_diff)
    fps_start = fps_end

    #print('FPS {}'.format(fps))

    key = cv2.waitKey(1)
    # press q to quit
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
    elif key == ord('f'):
        cv2.imwrite('positive/{}.png'.format(int(time())), screenshot)
    elif key == ord('d'):
        cv2.imwrite('negative/{}.png'.format(int(time())), screenshot)
