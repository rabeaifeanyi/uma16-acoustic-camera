import sounddevice as sd # type: ignore


def uma16_index():
    """Get the index of the UMA-16 microphone array.
    """
    devices = sd.query_devices()

    for index, device in enumerate(devices):
        if "nanoSHARC micArray16 UAC2.0" in device['name']:
            device_index = index
            print(f"\nUMA-16 device: {device['name']} at index {device_index}\n")
            break
    else:
        print("Could not find the UMA-16 device. Defaulting to the first device")
        device_index = 0 # Default to the first device
        
    return device_index

