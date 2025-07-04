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

# Function to read the telemetry data
def get_ets2_telemetry_data():
    shared_mem = None
    telemetry_data = {}

    try:
        shared_mem = mmap.mmap(-1, ETS2_PLUGIN_MMF_SIZE, ETS2_PLUGIN_MMF_NAME, access=mmap.ACCESS_READ)
        print(f"Successfully opened MMF: {ETS2_PLUGIN_MMF_NAME}")

        # --- List of (name, offset, type) ---
        fields = [
            ("time", 0, "uint32"),
            ("paused", 4, "uint32"),
            ("speed", 24, "float"),
            ("accelerationX", 28, "float"),
            ("accelerationY", 32, "float"),
            ("accelerationZ", 36, "float"),
            ("coordinateX", 40, "float"),
            ("coordinateY", 44, "float"),
            ("coordinateZ", 48, "float"),
            ("rotationX", 52, "float"),
            ("rotationY", 56, "float"),
            ("rotationZ", 60, "float"),
            ("gear", 64, "int32"),
            ("engineRpm", 80, "float"),
            ("fuel", 88, "float"),
            ("userSteer", 104, "float"),
            ("userThrottle", 108, "float"),
            ("userBrake", 112, "float"),
            ("userClutch", 116, "float"),
            ("truckWeight", 136, "float"),
            ("trailerWeight", 140, "float"),
            ("jobIncome", 300, "int32"),
            ("fuelRange", 880, "float"),
        ]

        def read_field(raw, offset, typ):
            if typ == "uint32":
                return get_uint32(raw, offset)
            elif typ == "int32":
                return get_int32(raw, offset)
            elif typ == "float":
                return get_float(raw, offset)
            elif typ == "byte":
                return raw[offset]
            elif typ.startswith("bytes["):
                length = int(typ[6:-1])
                return get_char_array(raw, offset, length)
            elif typ.startswith("float["):
                length = int(typ[6:-1])
                return [get_float(raw, offset + 4*i) for i in range(length)]
            else:
                return None

        # Read data from MMF
        shared_mem.seek(0)
        raw = shared_mem.read(ETS2_PLUGIN_MMF_SIZE)

        for name, offset, typ in fields:
            val = read_field(raw, offset, typ)
            telemetry_data[name] = val

        return telemetry_data

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

# Example usage
if __name__ == "__main__":
    data = get_ets2_telemetry_data()
    print(data)
