import asyncio 
import threading
import queue
import time
import csv
from datetime import datetime

import matplotlib.pyplot as plt
from bleak import BleakScanner, BleakClient
import numpy as np  # for moving average

CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# Thread-safe queue to store data for plotting and CSV-writing
data_queue = queue.Queue()

# Event to signal when to stop BLE notifications
stop_event = threading.Event() 

# Name for CSV file
csv_filename = f'adc_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

# Moving average window (in number of samples)
MOVING_AVG_WINDOW = 20  # adjust as you like


def notification_handler(sender, data):
    """
    This callback executes in Bleak's event loop (the separate thread).
    It parses data and places it into the queue. The main thread will
    handle CSV-writing and plotting.
    """
    # Assume the data is a comma-separated string, e.g. "-1.00, 2.50, 3.00, ..."
    decoded_data = data.decode().split(',')
    timestamp = datetime.now()

    try:
        # Parse each value as a float
        values = [float(x) for x in decoded_data]
    except ValueError as e:
        # If there's any invalid float, skip and log the error
        print(f"Skipping invalid data: {decoded_data}, error: {e} - Only one channel.py:42")
        return

    # Put data (timestamp + values) into the queue for the main thread
    data_queue.put((timestamp, values))


async def bleak_main():
    """
    - Discover and connect to the BLE device
    - Start notifications
    - Remain active until stop_event is signaled
    """
    print("Scanning for BLE devices... - Only one channel.py:55")
    devices = await BleakScanner.discover()
    device = next((d for d in devices if d.name and "bluetoothterminal" in d.name.lower()), None)

    if not device:
        print("Device not found. Exiting BLE thread. - Only one channel.py:60")
        return

    print(f"Found device: {device.name} ({device.address}). Attempting to connect... - Only one channel.py:63")
    async with BleakClient(device.address) as client:
        print(f"Connected to {device.name}! - Only one channel.py:65")

        # Start BLE notifications
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        print("Notifications started. Waiting for stop_event... - Only one channel.py:69")

        # Keep the async loop alive until we set stop_event
        while not stop_event.is_set():
            await asyncio.sleep(0.1)

        # Optionally stop notifications if desired
        await client.stop_notify(CHARACTERISTIC_UUID)
        print("Stopped notifications. - Only one channel.py:77")

    print("BLE thread exiting. - Only one channel.py:79")


def run_bleak_loop():
    """Target for the separate thread to run BLE asynchronously."""
    asyncio.run(bleak_main())


def smooth_with_moving_average(channel_vals: np.ndarray, window: int) -> np.ndarray:
    """
    Moving average with edge padding that always returns an array
    of the same length as channel_vals, for any window size.
    """
    n = len(channel_vals)
    if window <= 1 or n < window:
        # No smoothing if window too small or not enough samples
        return channel_vals

    # Window-1 total padding, split left/right
    total_pad = window - 1
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left

    padded = np.pad(channel_vals, (pad_left, pad_right), mode='edge')
    kernel = np.ones(window) / window

    # 'valid' length = n
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed


if __name__ == "__main__":
    # Start the BLE thread
    ble_thread = threading.Thread(target=run_bleak_loop, daemon=True)
    ble_thread.start()

    # Channels to exclude from plotting (0-indexed)
    # Here: only channel index 5 (Channel 6) is plotted
    exclude_channels = {7,14,15,16,17,18,19,20}

    # --------------------
    # Set up real-time plot
    # --------------------
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    # We’ll initialize lines once we know the number of channels
    lines = []
    num_channels = 0
    plot_channels = []  # indices of channels we actually plot

    # A buffer to hold the most recent data for plotting
    data_buffer = []
    buffer_size = 2000  # keep up to 2000 data points

    # Track whether we have written the CSV header yet
    csv_header_written = False

    # For measuring relative time on the X-axis
    first_timestamp = None

    try:
        while True:
            # Retrieve any new data from the queue
            while not data_queue.empty():
                timestamp, values = data_queue.get()

                # If this is the first data, set up channels
                if num_channels == 0:
                    num_channels = len(values)
                    # Decide which channels to plot (one or more)
                    plot_channels = [i for i in range(num_channels) if i not in exclude_channels]

                # If we haven't set up the plot lines yet, do it now
                if not lines and num_channels > 0:
                    for i in plot_channels:
                        line, = ax.plot([], [], label=f'Channel {i+1}')
                        lines.append(line)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('ADC Value')
                    ax.set_title('Real-time ADC Data (Moving Average)')
                    ax.legend()

                # If we haven't written the CSV header, do it now
                if not csv_header_written and plot_channels:
                    with open(csv_filename, 'w', newline='') as file:
                        writer = csv.writer(file)
                        # Only save timestamp + plotted channels
                        header = ['Timestamp'] + [f'Channel_{i+1}' for i in plot_channels]
                        writer.writerow(header)
                    csv_header_written = True

                # Append data to buffer
                if first_timestamp is None:
                    first_timestamp = timestamp

                data_buffer.append((timestamp, values))
                if len(data_buffer) > buffer_size:
                    data_buffer.pop(0)

                # Write this data row to CSV (append mode) - ONLY plotted channels
                if csv_header_written:
                    with open(csv_filename, 'a', newline='') as file:
                        writer = csv.writer(file)
                        row_values = [values[i] for i in plot_channels]
                        writer.writerow([timestamp] + row_values)

            # Update the plot if we have data
            if data_buffer and lines:
                # Generate array of times in seconds relative to the first timestamp
                times = [(dp[0] - first_timestamp).total_seconds() for dp in data_buffer]
                times_arr = np.array(times)

                # Update each plotted channel’s line data with MOVING AVERAGE
                for line, ch_idx in zip(lines, plot_channels):
                    channel_vals = np.array([dp[1][ch_idx] for dp in data_buffer], dtype=float)
                    smoothed = smooth_with_moving_average(channel_vals, MOVING_AVG_WINDOW)
                    line.set_data(times_arr, smoothed)

                ax.relim()
                ax.autoscale_view()

            # Redraw the figure
            fig.canvas.draw()
            fig.canvas.flush_events()

            # If the figure is closed, break
            if not plt.fignum_exists(fig.number):
                break

            # Short pause to avoid maxing out CPU
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting... - Only one channel.py:213")

    finally:
        # Signal the BLE thread to stop
        stop_event.set()
        ble_thread.join()

        # Clean up the plot
        plt.close(fig)
        print("Done. - Only one channel.py:222")
