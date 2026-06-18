"""Quick BLE scan — lists all nearby devices so you can find the exact name."""
import asyncio
from bleak import BleakScanner

async def scan():
    print("Scanning for 10 seconds...\n")
    devices = await BleakScanner.discover(timeout=10)
    if not devices:
        print("No BLE devices found. Is Bluetooth enabled?")
        return
    print(f"Found {len(devices)} device(s):\n")
    for d in sorted(devices, key=lambda x: x.name or ""):
        print(f"  Name: {repr(d.name):<30}  Address: {d.address}")

asyncio.run(scan())
