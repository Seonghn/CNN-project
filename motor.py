import RPi.GPIO as GPIO
import time
import sys

GPIO.setmode(GPIO.BOARD)

DCpin1 = 32
DCpin2 = 33

GPIO.setup(DCpin1, GPIO.OUT)
GPIO.setup(DCpin2, GPIO.OUT)

pwm1 = GPIO.PWM(DCpin1, 100)
pwm2 = GPIO.PWM(DCpin2, 100)
pwm1.start(0)
pwm2.start(0)

def Foward(speed):
    pwm1.ChangeDutyCycle(speed)

def Backward(speed):
    pwm2.ChangeDutyCycle(spped)

def Stop():
        pwm1.stop()
        pwm1.stop()

try:
    while True:
        Foward(20)
        time.sleep(1)
        Stop()

except KeyboardInterrupt:
    pwm1.stop()
    GPIO.cleanup()
    sys.exit()
