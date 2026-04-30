import usb.core
import usb.util
import time

VID = 0x1074
PID = 0x0005

dev = usb.core.find(idVendor=VID, idProduct=PID)

if dev is None:
    raise ValueError("Device not found")

print("Device ready")

dev.set_configuration()

# detach kernel driver if needed
try:
    dev.detach_kernel_driver(0)
except:
    pass


# =========================
# CONTROL TRANSFER TEST
# =========================
def ctrl_out(req, val, idx, data):
    return dev.ctrl_transfer(
        0x40,  # Host → Device
        req,
        val,
        idx,
        data
    )

def ctrl_in(req, val, idx, length):
    return dev.ctrl_transfer(
        0xC0,  # Device → Host
        req,
        val,
        idx,
        length
    )


print("\n🔍 Trying control init...")

# brute try common init patterns
init_cmds = [
    (0x01, 0x0001, 0x0000, [1]),
    (0x01, 0x0000, 0x0000, [1]),
    (0x02, 0x0001, 0x0000, [0]),
    (0x09, 0x0001, 0x0000, [1]),
    (0x0A, 0x0001, 0x0000, [1]),
]

for req, val, idx, data in init_cmds:
    try:
        print(f"Trying CTRL {req}")
        ctrl_out(req, val, idx, data)
        time.sleep(0.2)
    except Exception as e:
        print("Fail:", e)


# =========================
# STREAM READ TEST
# =========================
cfg = dev.get_active_configuration()
intf = cfg[(0, 0)]

# pick biggest IN endpoint
ep_in = max(
    [e for e in intf if e.bEndpointAddress & 0x80],
    key=lambda e: e.wMaxPacketSize
)

print(f"\nUsing endpoint: {hex(ep_in.bEndpointAddress)}")

print("Streaming...")

last = None

while True:
    try:
        data = dev.read(ep_in.bEndpointAddress, 512, timeout=200)

        # detect change
        h = hash(bytes(data))

        if last != h:
            print("NEW DATA:", data[:16])
            last = h

    except Exception as e:
        pass