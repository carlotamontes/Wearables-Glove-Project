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
    decoded_data = data.decode().split(',')
    timestamp = datetime.now()

    try:
        values = [float(x) for x in decoded_data]
    except ValueError as e:
        print(f"Skipping invalid data: {decoded_data}, error: {e}")
        return

    print(f"[BLE] Received {len(values)} values: {values[:5]}...")
    data_queue.put((timestamp, values))


async def bleak_main():
    """
    - Discover and connect to the BLE device
    - Start notifications
    - Remain active until stop_event is signaled
    """
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover()
    device = next((d for d in devices if d.name and "bluetoothterminal" in d.name.lower()), None)

    if not device:
        print("Device not found. Exiting BLE thread.")
        return

    print(f"Found device: {device.name} ({device.address}). Attempting to connect...")
    async with BleakClient(device.address) as client:
        print(f"Connected to {device.name}!")

        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        print("Notifications started. Waiting for stop_event...")

        while not stop_event.is_set():
            await asyncio.sleep(0.1)

        await client.stop_notify(CHARACTERISTIC_UUID)
        print("Stopped notifications.")

    print("BLE thread exiting.")


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
        return channel_vals

    total_pad = window - 1
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left

    padded = np.pad(channel_vals, (pad_left, pad_right), mode='edge')
    kernel = np.ones(window) / window

    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed


if __name__ == "__main__":
    # Start the BLE thread
    ble_thread = threading.Thread(target=run_bleak_loop, daemon=True)
    ble_thread.start()

    # -----------------------------------------------------------------
    # Channels we care about (0-indexed), corresponding to
    # Arduino pins 1, 4, 6, 11, 13 -> python indices 0, 3, 5, 10, 12
    # (resistances at rest: ~200k, 50k, 50k, 50k, 20k)
    # -----------------------------------------------------------------
    plot_channels = [0, 3, 5, 10, 12]

    # Optional: human-readable labels per channel (Arduino pin numbers)
    channel_labels = {
        0: "Channel 1 (pin 1, ~200k)",
        3: "Channel 4 (pin 4, ~50k)",
        5: "Channel 6 (pin 6, ~50k)",
        10: "Channel 11 (pin 11, ~50k)",
        12: "Channel 13 (pin 13, ~20k)",
    }

    num_plot_channels = len(plot_channels)

    # --------------------
    # Set up real-time plot: one window, 5 subplots stacked vertically
    # --------------------
    plt.ion()
    fig, axes = plt.subplots(num_plot_channels, 1, figsize=(10, 2.5 * num_plot_channels), sharex=True)

    if num_plot_channels == 1:
        axes = [axes]

    lines = []
    for ax, ch_idx in zip(axes, plot_channels):
        line, = ax.plot([], [], label=channel_labels.get(ch_idx, f"Channel {ch_idx+1}"))
        lines.append(line)
        ax.set_ylabel("ADC Value")
        ax.set_title(channel_labels.get(ch_idx, f"Channel {ch_idx+1}"))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Real-time Flex Sensor Data (Moving Average) - 5 individual plots")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    num_channels = 0  # total channels in incoming data (set on first message)

    # A buffer to hold the most recent data for plotting
    data_buffer = []
    buffer_size = 2000  # keep up to 2000 data points

    csv_header_written = False
    first_timestamp = None

    try:
        while True:
            # Retrieve any new data from the queue
            while not data_queue.empty():
                timestamp, values = data_queue.get()

                if num_channels == 0:
                    num_channels = len(values)
                    print(f"Detected {num_channels} channels in incoming data.")

                # Write CSV header once
                if not csv_header_written:
                    with open(csv_filename, 'w', newline='') as file:
                        writer = csv.writer(file)
                        header = ['Timestamp'] + [f'Channel_{i+1}' for i in plot_channels]
                        writer.writerow(header)
                    csv_header_written = True

                if first_timestamp is None:
                    first_timestamp = timestamp

                data_buffer.append((timestamp, values))
                if len(data_buffer) > buffer_size:
                    data_buffer.pop(0)

                # Append row to CSV (only the channels we care about)
                with open(csv_filename, 'a', newline='') as file:
                    writer = csv.writer(file)
                    row_values = [values[i] for i in plot_channels]
                    writer.writerow([timestamp] + row_values)

            # Update the plots if we have data
            if data_buffer:
                times = [(dp[0] - first_timestamp).total_seconds() for dp in data_buffer]
                times_arr = np.array(times)

                for line, ax, ch_idx in zip(lines, axes, plot_channels):
                    channel_vals = np.array([dp[1][ch_idx] for dp in data_buffer], dtype=float)
                    smoothed = smooth_with_moving_average(channel_vals, MOVING_AVG_WINDOW)
                    line.set_data(times_arr, smoothed)

                    # Each subplot autoscales independently -> different
                    # magnitudes (18k vs 200k) are all clearly visible
                    ax.relim()
                    ax.autoscale_view()

                print(f"[PLOT] buffer={len(data_buffer)} points, last values: {data_buffer[-1][1][:5]}")

            fig.canvas.draw()
            fig.canvas.flush_events()

            if not plt.fignum_exists(fig.number):
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")

    finally:
        stop_event.set()
        ble_thread.join()

        plt.close(fig)
        print("Done.")