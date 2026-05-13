import logging
import time
import numpy as np
import serial
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class InvalidOperationError(Exception):
    pass

logger = logging.getLogger(__name__)


class VectorNetworkAnalyzer:
    name = "vector network analyzer"
    abbreviation = "VNA"
    extended_name = f"{name} ({abbreviation})"

    def __init__(self):

        #Serial Communication Protocol
        self.ser = serial.Serial()
        self.ser.baudrate = 115200
        self.ser.timeout = 1

        #Yet to be implemented
        self.context: np.ndarray | None = None

        self.normalized_reflection_1darray: np.ndarray | None = None
        self.normalized_transmission_1darray: np.ndarray | None = None

    def connect(self, port:str):
        self.ser.port = port
        if self.is_connected:
            raise InvalidOperationError(f"{self.name.capitalize()} is already connected.")
        self.ser.open()
        logger.info(f"{self.name.capitalize()} connected")

    def disconnect(self):
        if not self.is_connected:
            raise InvalidOperationError(f"{self.name.capitalize()} is already disconnected.")
        self.ser.close()
        logger.info(f"{self.name.capitalize()} disconnected")

    @property
    def is_connected(self) -> bool:
        return self.ser.isOpen()

    def __run_command(self,
                      command,
                      dtype: None |
                             type[str] |
                             type[list[str]] |
                             type[list[list[float]]] = str) -> (None |
                                                                str |
                                                                list[str] |
                                                                list[list[float]]):
        command += "\r"
        if self.ser.in_waiting > 0:
            logger.debug(f"{self.ser.in_waiting=}")
            logger.debug(f"{self.ser.read(self.ser.in_waiting).decode('utf-8')=}")
            self.ser.reset_input_buffer()
        if self.ser.out_waiting > 0:
            logger.debug(f"{self.ser.out_waiting=}")
            self.ser.reset_output_buffer()

        self.ser.write("pause\r".encode("utf-8"))
        self.ser.write("resume\r".encode("utf-8"))

        #Initialization of VNA may take some time, so we need to wait for a while before sending the command
        #From experimentation, it is possible so set delay to 1 second. However, 2 seconds is safer to avoid any potential issues.
        time.sleep(2)

        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        self.ser.write(command.encode("utf-8"))

        #Transmission of command from PC to VNA may take some time, so we need to wait for a while before reading the response
        #From experimentation, it is possible so set delay to 1 second. However, 2 seconds is safer to avoid any potential issues.
        time.sleep(2)

        if dtype is None:
            return None

        response_str: str = ""
        try:
            while self.ser.in_waiting > 0:
                response_bytes: bytes = self.ser.read(self.ser.in_waiting)
                response_str += response_bytes.decode("utf-8")

                #Transmission of data from VNA to PC may take some time, so we need to wait for a while before reading the response
                time.sleep(3)

        except Exception as e:
            logger.exception(f"Failed to read the response from {self.name} ({e=}) ({command=})")

        lines: list[str] = response_str.strip().split("\r\n")
        lines = lines[1:len(lines) - 1]
        if dtype == str:
            return "\n".join(lines)
        if dtype == list[str]:
            return lines
        return [list(map(float, line.split())) for line in lines]

    def reset(self, should_reconnect_after_resetting: bool = True):
        self.__run_command("reset", None)
        time.sleep(10)
        self.disconnect()
        if should_reconnect_after_resetting:
            self.connect()

    def scan(self, frequency_start_Hz, frequency_stop_Hz, num_of_points, outmask: int = 6) -> list[list[float]]:
        print(frequency_start_Hz, frequency_stop_Hz, num_of_points)
        logger.info(
            f"Scanning from {int(frequency_start_Hz)}Hz to {int(frequency_stop_Hz)}Hz with {int(num_of_points)} points",
            extra={
                "frequency_start_Hz": int(frequency_start_Hz),
                "frequency_stop_Hz": int(frequency_stop_Hz),
                "num_of_points": int(num_of_points),
            }
        )
        command = f"scan {int(frequency_start_Hz)} {int(frequency_stop_Hz)} {int(num_of_points)} {outmask}"
        return self.__run_command(command, list[list[float]])

    def beep(self, is_on: bool = True) -> None:
        command = "beep "
        if is_on:
            command += "on"
        else:
            command += "off"
        self.__run_command(command, None)

    @property
    def version(self) -> str:
        return self.__run_command("version")

    @property
    def info(self) -> str:
        return self.__run_command("info")

    @property
    def SN(self) -> str:
        return self.__run_command("SN")

    def measure_s_parameter(self, frequency_start, frequency_stop, num_of_points: int,
                            is_for_preparation: bool = False):
        print(frequency_start, frequency_stop, num_of_points)
        response = self.scan(frequency_start,  # Unit: Hz
                             frequency_stop,  # Unit: Hz
                             num_of_points)
        self.frequency_Hz_1darray: np.ndarray = np.linspace(frequency_start, frequency_stop, num_of_points)  # Unit: Hz
        self.frequency_GHz_1darray: np.ndarray = self.frequency_Hz_1darray / 1000000000  # Unit: GHz

        data_array: np.ndarray = np.array(response)
        
        self.data_temp = data_array

        # Extract y1 and y2 of S11: Reflection coefficient
        # Column 0: Real part of y1
        self.S11_real_1darray: np.ndarray = data_array[:, 0]
        # Column 1: Imaginary part of y1
        self.S11_imag_1darray: np.ndarray = data_array[:, 1]

        # Extract y1 and y2 of S21: Transmission coefficient
        # Column 2: Real part of y1
        self.S21_real_1darray: np.ndarray = data_array[:, 2]
        # Column 3: Imaginary part of y1
        self.S21_imag_1darray: np.ndarray = data_array[:, 3]

        self.S11_1darray: np.ndarray = self.S11_real_1darray + 1j * self.S11_imag_1darray
        self.reflection_1darray: np.ndarray = abs(self.S11_1darray)
        self.phase11_1darray: np.ndarray = np.angle(self.S11_1darray)
        self.mag11_avg_1darray: np.ndarray = ensemble_average(self.reflection_1darray, wd=5)
        self.phase11_avg_1darray: np.ndarray = ensemble_average(self.phase11_1darray, wd=5)

        self.S21_1darray: np.ndarray = self.S21_real_1darray + 1j * self.S21_imag_1darray
        self.transmission_1darray: np.ndarray = abs(self.S21_1darray)  # magnitude of `self.S21_1darray`
        self.phase21_1darray: np.ndarray = np.angle(self.S21_1darray)
        self.mag21_avg_1darray: np.ndarray = ensemble_average(self.transmission_1darray, wd=5)
        self.phase21_avg_1darray: np.ndarray = ensemble_average(self.phase21_1darray, wd=5)


        if is_for_preparation:
            self.frequency_Hz_1darray_for_normalization = self.frequency_Hz_1darray
            self.S11_1darray_for_normalization = self.S11_1darray
            self.S21_1darray_for_normalization = self.S21_1darray
            self.f_for_S21_1darray_normalization = interp1d(self.frequency_Hz_1darray_for_normalization,
                                                                    self.S21_1darray_for_normalization,
                                                                    kind="cubic",
                                                                    bounds_error=False,
                                                                    fill_value="extrapolate")
        if self.f_for_S21_1darray_normalization is not None:
            normalized_S21_1darray: np.ndarray = self.S21_1darray / self.f_for_S21_1darray_normalization(
                self.frequency_Hz_1darray)
            self.normalized_transmission_1darray = abs(normalized_S21_1darray)


def ensemble_average(data, wd=5):
    return np.convolve(data, np.ones(wd) / wd, mode="valid")


"""""""""""""""""""""""""""
The following code would only run when this file is executed directly.
It should produce the spectrum scan plot and print the peak frequency in the console.
To stop the code, simply press Ctrl + C in the console.
"""""""""""""""""""""""""""

if __name__ == "__main__":

    #Starting VNA connection
    test = VectorNetworkAnalyzer()
    test.connect("COM3")
    
    #Normalisation run
    test.measure_s_parameter(2000000000, 3000000000, 150, is_for_preparation=True)

    try:
        while True:

            #Take measurment
            STtime = int(time.time())
            test.measure_s_parameter(2000000000, 3000000000, 150, is_for_preparation=False)
            Pass_time = int(time.time()) - STtime
            print(f"Measurment Time = {Pass_time}")
    
            #Plotting and peak finding
            plt.figure(1)
            plt.plot(test.frequency_GHz_1darray, test.normalized_transmission_1darray**-1)
            peaks = find_peaks(test.normalized_transmission_1darray**-1, height=1.1)[0]

            if len(peaks) == 0:
                print("No peaks found")
            else:
                print(peaks)
                peaks = np.max(peaks)
                print(peaks)
                print(f"Peak found at {test.frequency_GHz_1darray[peaks]} GHz")
                
            plt.show()
            
    except KeyboardInterrupt:
        test.disconnect()

