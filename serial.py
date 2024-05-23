import serial

# Configure the serial port
ser = serial.Serial('/dev/ttyAMA0', 9600)  # Change baudrate if necessary

try:
    while True:
        # Read a line of data from the serial port
        line = ser.readline().decode('utf-8').strip()
        print(line)  # Print the received data
except KeyboardInterrupt:
    ser.close()  # Close the serial port when Ctrl+C is pressed
