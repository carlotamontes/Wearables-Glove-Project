"""Test full BLE connection to group8 and print incoming data."""
import asyncio
from bleak import BleakScanner, BleakClient

DEVICE_NAME = "group8"
CHAR_UUID   = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

def handler(sender, data):
    print(f"  Data: {data.decode().strip()}")

async def main():
    print("Scanning...")
    devices = await BleakScanner.discover(timeout=10)
    device = next((d for d in devices if d.name and DEVICE_NAME in d.name.lower()), None)

    if not device:
        print("NOT FOUND — is ESP32 on and not connected to another app?")
        return

    print(f"Found: {device.name} @ {device.address}")
    print("Connecting...")
    async with BleakClient(device.address) as client:
        print(f"Connected! Listening for 5 seconds...")
        await client.start_notify(CHAR_UUID, handler)
        await asyncio.sleep(5)
        await client.stop_notify(CHAR_UUID)
    print("Done.")

asyncio.run(main())
