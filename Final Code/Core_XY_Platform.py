import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from pathlib import Path
from scipy.stats import norm

import sys
import time
from telemetrix import telemetrix


class CoreXY_Platform:
    def __init__(self):
        # IO PIN Definitions
        self.PULSE_PIN0 = 6
        self.DIRECTION_PIN0 = 7
        self.PULSE_PIN1 = 8
        self.DIRECTION_PIN1 = 9
        self.EN_PIN = 10
        self.HM_PIN = 3
        self.exit_flag = 0
        self.HM_Stats = 0

        # Assign to object
        print('Initializing Board')
        self.board = telemetrix.Telemetrix()
        print('Board connected and ready')
        print('Setting up Pin')
        self.motor0 = self.board.set_pin_mode_stepper(interface=1, pin1=self.PULSE_PIN0, pin2=self.DIRECTION_PIN0) #Motor 1 on driver board
        self.motor1 = self.board.set_pin_mode_stepper(interface=1, pin1=self.PULSE_PIN1, pin2=self.DIRECTION_PIN1) #Motor 2 on driver board
        self.EN_Pin = self.board.set_pin_mode_digital_output(self.EN_PIN)
        self.HM_Pin = self.board.set_pin_mode_digital_input_pullup(self.HM_PIN, callback=self.HM_callback)
        time.sleep(.5)
        self.Set_Max_Speed_Accel(Max_Speed=1000, Accel=1000)

    # Relative Homing    
    def Manuel_Home(self):
        self.board.digital_write(self.EN_PIN, 0) # Disable stepper
        print('Stepper Disabled')
        input("Move Header to Home Position and press Enter")
        self.board.digital_write(self.EN_PIN, 1) # Enable stepper
        print('Stepper Enabled')
        self.board.stepper_set_current_position(0, 0) # Set current position to 0


    # Absolute Homing with sensor check
    def Manuel_Home_Absolute(self):
        self.board.digital_write(self.EN_PIN, 0) # Disable stepper
        print('Stepper Disabled')
        #Condition check for home position
        while True:
            input("Move Header to Home Position and press Enter")
            if self.HM_Stats == 1:
                break
            else:
                print('Home position not detected, please try again.')
                time.sleep(1)

        self.board.digital_write(self.EN_PIN, 1) # Enable stepper
        print('Stepper Enabled')
        self.board.stepper_set_current_position(0, 0) # Set current position to 0

    # Sensor callback    
    def HM_callback(self, data):
        self.HM_Stats = data[2]                                         

    # Set Max Speed and Acceleration    
    def Set_Max_Speed_Accel(self, Max_Speed = 1000, Accel = 1000):
        print(f'Setting Max Speed to {Max_Speed} steps/sec and Acceleration to {Accel} steps/sec^2')
        self.board.stepper_set_max_speed(self.motor0, Max_Speed)
        self.board.stepper_set_acceleration(self.motor0, Accel)
        self.board.stepper_set_max_speed(self.motor1, Max_Speed)
        self.board.stepper_set_acceleration(self.motor1, Accel)

    # Callback for motor movement completion
    def the_callback(self, data):
        date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data[2]))
        print(f'Motor {data[1]} absolute motion completed at: {date}.')
        self.exit_flag += 1

    # Shutdown platform        
    def Shutdown(self):
        print('Shutting down platform')
        self.board.digital_write(self.EN_PIN, 0)
        self.board.shutdown()

    # Move to Position in mm   
    def Move_to_Pos(self, X_Pos, Y_Pos):
        # Check position limits
        if X_Pos > 66 or Y_Pos > 80:
            print('Position out of range, Max X = 66mm, Max Y = 80mm')
            self.Shutdown()
            raise ValueError('Position out of range')
        # Reset exit flag
        self.exit_flag = 0
        # Invert Y movement direction
        Y_Pos = -Y_Pos
        # Coordinate conversion to steps
        Steps_per_mm = 99.65
        # Motor0 = Left Motor, Motor1 = Right Motor
        #Cartesian to CoreXY conversion
        Motor0_steps = int((X_Pos - Y_Pos) * Steps_per_mm)
        Motor1_steps = int((X_Pos + Y_Pos) * Steps_per_mm)
        # Set position to move to
        self.board.stepper_move_to(self.motor0, Motor0_steps)
        self.board.stepper_move_to(self.motor1, Motor1_steps)
        time.sleep(.1)
        #Run motors
        self.board.stepper_run(self.motor0, completion_callback=self.the_callback)
        self.board.stepper_run(self.motor1, completion_callback=self.the_callback)   
        time.sleep(.1)
        # wait until both motors have finished
        while self.exit_flag < 2:
            try:    
                time.sleep(.1)
            except KeyboardInterrupt:
                print('Shutting down platform')
                self.board.digital_write(self.EN_PIN, 0)
                self.board.shutdown()


"""
The following code would only execute if this script is run directly, and not imported as a module.
It initializes the CoreXY platform, then move in a triangle pattern for testing.
"""

if __name__ == "__main__":
    Platform = CoreXY_Platform()
    Platform.Manuel_Home_Absolute()
    Platform.Set_Max_Speed_Accel(1000, 1000)
    Platform.Move_to_Pos(21,1)

    input("Temp Stop")

    Platform.Move_to_Pos(0,5)
    input('Carrage moved to (0,5), press Enter to continue')
    Platform.Move_to_Pos(5,5)
    input('Carrage moved to (5,5), press Enter to continue')
    Platform.Move_to_Pos(0,0)
    input('Carrage moved to (0,0), press Enter to continue')
    Platform.Shutdown()







