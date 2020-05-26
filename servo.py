import RPi.GPIO as GPIO
import time

pin =18
GPIO.setmode(GPIO.BCM)
GPIO.setup(pin, GPIO.OUT)
p= GPIO.PWM(pin, 50)  #PMW:ÆÞ½º Æø º¯Á¶
p.start(0)
cnt = 0

try:

    while True:
        p.ChangeDutyCycle(12.5) #ÃÖ´ñ°ª
        time.sleep(1)
        p.ChangeDutyCycle(10.0)
        time.sleep(1)
        p.ChangeDutyCycle(7.5) #0
        time.sleep(1)
        p.ChangeDutyCycle(5.0)
        time.sleep(1)
        p.ChangeDutyCycle(2.5) #ÃÖ¼Ú°ª
        time.sleep(1)

except KeybordInterrupt:
     p.stop()
    
GPIO.cleanup()