import mmap
import ctypes
import time
import traceback

ETS2_PLUGIN_MMF_NAME = "Local\\SimTelemetryETS2"
ETS2_PLUGIN_MMF_SIZE = 16 * 1024

# Offsets for fields in the MMF (from ets2-telemetry-common.hpp)
# These offsets are for plugin revision 5, SDK 1.4/1.5 (nlhans/ets2-sdk-plugin)
# You may need to adjust if using a different version.

# Helper: get offset of a field in a ctypes struct
def field_offset(struct, field):
    return getattr(struct, field).offset

# Define the full struct as a 16KB byte array
class Ets2TelemetryRaw(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_ubyte * ETS2_PLUGIN_MMF_SIZE)]

def get_float(raw, offset):
    return ctypes.c_float.from_buffer_copy(raw[offset:offset+4]).value

def get_uint32(raw, offset):
    return ctypes.c_uint32.from_buffer_copy(raw[offset:offset+4]).value

def get_bool(raw, offset):
    return bool(ctypes.c_ubyte.from_buffer_copy(raw[offset:offset+1]).value)

def get_int32(raw, offset):
    return ctypes.c_int32.from_buffer_copy(raw[offset:offset+4]).value

def get_char_array(raw, offset, length):
    return bytes(raw[offset:offset+length]).split(b'\x00', 1)[0].decode('utf-8', errors='ignore')

def print_floats(raw, start, count):
    floats = []
    for i in range(count):
        offset = start + i * 4
        val = get_float(raw, offset)
        floats.append(f"{val:.3f}")
    return floats

def print_floats_block(raw, start, count):
    floats = []
    for i in range(count):
        offset = start + i * 4
        val = get_float(raw, offset)
        floats.append(val)
    return floats

def main():
    shared_mem = None
    try:
        shared_mem = mmap.mmap(-1, ETS2_PLUGIN_MMF_SIZE, ETS2_PLUGIN_MMF_NAME, access=mmap.ACCESS_READ)
        print(f"Successfully opened MMF: {ETS2_PLUGIN_MMF_NAME}")
        zero_count = 0
        for _ in range(5):
            shared_mem.seek(0)
            raw = shared_mem.read(ETS2_PLUGIN_MMF_SIZE)
            if all(b == 0 for b in raw):
                print("All MMF bytes are zero. The plugin is NOT writing data.")
                zero_count += 1
                time.sleep(1)
            else:
                print("MMF is not all zero. The plugin is writing data.")
                break
        if zero_count == 5:
            print("\nERROR: The plugin is not writing to the MMF.\n"
                  "Check the following:\n"
                  "1. Plugin DLL is in the correct plugins folder for your ETS2 bitness (win_x86 or win_x64).\n"
                  "2. You see the SDK activation popup when starting ETS2.\n"
                  "3. The plugin DLL is not blocked by Windows (right-click > Properties > Unblock).\n"
                  "4. Restart ETS2 after copying the plugin.\n"
                  "5. Try the C# demo client from the SDK to verify plugin/MMF operation.\n"
                  "Your Python code is correct, but the plugin is not active or not compatible with your game version.")
            return

        last_time = -1

        TEL_REV3_OFFSET = 0x1A0
        # Damage offsets found by scanning
        DAMAGE_ENGINE_OFFSET = TEL_REV3_OFFSET + 4 * 57
        DAMAGE_TRANSMISSION_OFFSET = TEL_REV3_OFFSET + 4 * 58
        DAMAGE_CABIN_OFFSET = TEL_REV3_OFFSET + 4 * 59
        DAMAGE_CHASSIS_OFFSET = TEL_REV3_OFFSET + 4 * 60

        # Offsets for speed and turn radius (adjust according to the SDK documentation)
        SPEED_OFFSET = TEL_REV3_OFFSET + 4 * 1  # Example offset for speed
        TURN_RADIUS_OFFSET = TEL_REV3_OFFSET + 4 * 21  # Example offset for turn radius

        while True:
            shared_mem.seek(0)
            raw = shared_mem.read(ETS2_PLUGIN_MMF_SIZE)
            time_val = get_uint32(raw, 0)

            engine_damage = get_float(raw, DAMAGE_ENGINE_OFFSET)
            transmission_damage = get_float(raw, DAMAGE_TRANSMISSION_OFFSET)
            cabin_damage = get_float(raw, DAMAGE_CABIN_OFFSET)
            chassis_damage = get_float(raw, DAMAGE_CHASSIS_OFFSET)

            # Read speed and turn radius
            speed = get_float(raw, SPEED_OFFSET)
            turn_radius = get_float(raw, TURN_RADIUS_OFFSET)

            print("-" * 20)
            print(f"Timestamp: {time_val}")
            print(f"Engine Damage: {engine_damage:.3f} ({engine_damage*100:.1f}%)")
            print(f"Transmission Damage: {transmission_damage:.3f} ({transmission_damage*100:.1f}%)")
            print(f"Cabin Damage: {cabin_damage:.3f} ({cabin_damage*100:.1f}%)")
            print(f"Chassis Damage: {chassis_damage:.3f} ({chassis_damage*100:.1f}%)")
            print(f"Speed: {speed:.3f} km/h")
            print(f"Turn Radius: {turn_radius:.3f} meters")
            print("Cause more damage in-game to verify which value is which.")

            time.sleep(1)
    except FileNotFoundError:
        print(f"Error: Memory-mapped file '{ETS2_PLUGIN_MMF_NAME}' not found.")
        print("Ensure ETS2/ATS is running with the Telemetry SDK Plugin installed and active.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if shared_mem:
            shared_mem.close()
            print("MMF closed.")

if __name__ == "__main__":
    main()