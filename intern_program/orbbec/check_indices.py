import pyorbbecsdk
from pyorbbecsdk import Context, DeviceList

def check():
    try:
        # Create the context
        ctx = pyorbbecsdk.Context()
        device_list = ctx.query_devices()
        
        count = device_list.get_count()
        print(f"Total devices found: {count}")
        
        for i in range(count):
            try:
                dev = device_list.get_device_by_index(i)
                info = dev.get_device_info()
                print(f"Index {i}: Name: {info.get_name()}, PID: {hex(info.get_pid())}")
            except Exception as e:
                print(f"Index {i} metadata access failed: {e}")
                
    except Exception as e:
        print(f"Failed to initialize Orbbec Context: {e}")

if __name__ == "__main__":
    check()