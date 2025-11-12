# serial_sender.py

import re
import time
import threading
from typing import Optional

import serial
import serial.tools.list_ports as list_ports


class SerialSender:
    """ 아두이노로 '세 자리 코드(예: 900, 500, 600...)' 전송 """
    def __init__(self, port: Optional[str], baud: int, dup_policy: str = "handshake", disable_serial: bool = False):
        self.port_name = find_serial_port(port)
        self.baud = baud
        self.ser = None
        self.last_sent = None 
        self.dup_policy = dup_policy
        self._stop_reader = False
        self._reader_th: Optional[threading.Thread] = None

        if disable_serial:
            print("[INFO] Serial communication is disabled by --disable_serial.")
            return

        if self.port_name:
            try:
                self.ser = serial.Serial(self.port_name, self.baud, timeout=0.1)
                time.sleep(1.5)  # Uno 리셋 대기
                print(f"[INFO] Serial opened: {self.port_name} @ {self.baud}")
                if self.dup_policy == "handshake":
                    self._reader_th = threading.Thread(target=self._reader_loop, daemon=True)
                    self._reader_th.start()
            except Exception as e:
                print(f"[WARN] Serial open failed ({self.port_name}): {e}")
                self.ser = None
        else:
            print("[WARN] No serial port found. Running without sending.")

    def _reader_loop(self):
        while not self._stop_reader and self.ser is not None:
            try:
                line = self.ser.readline()
                if not line:
                    continue
                text = line.decode("utf-8", errors="ignore").strip()
                if text:
                    print(f"[SER] {text}")
                    if "READY" in text or "[STATE] IDLE" or "RoboSort Ready" in text:
                        self.last_sent = None
            except Exception:
                time.sleep(0.05)

    def send_code(self, code: int):
        """ 세 자리 정수 코드를 전송 """
        if code is None:
            return
        if self.ser is None:
            print(f"[SEND-DRY] {code}")
            return

        if self.dup_policy == "block" and self.last_sent == code:
            return
        if self.dup_policy == "handshake" and self.last_sent == code:
            return

        line = f"{int(code)}\n".encode()
        try:
            self.ser.write(line)
            self.ser.flush()
            print(f"[SEND] {line!r} -> {self.port_name}")
            self.last_sent = int(code)
        except Exception as e:
            print(f"[WARN] Serial send failed: {e}")
            self.ser = None

    def close(self):
        try:
            self._stop_reader = True
            if self._reader_th and self._reader_th.is_alive():
                self._reader_th.join(timeout=0.2)
            if self.ser: self.ser.close()
        except Exception:
            pass

# ===================== 구역/시리얼 유틸 =====================
def code_to_zone(code: str) -> Optional[int]:
    """ '619 A02' -> 600  (앞 3자리의 백의 자리로 내림 ×100) """
    m = re.search(r"(\d{3})\s?[A-Z]\d{2}", code.upper())
    if not m: return None
    abc = int(m.group(1))
    return (abc // 100) * 100

def find_serial_port(prefer: Optional[str]=None) -> Optional[str]:
    if prefer:
        return prefer
    ports = list_ports.comports()
    for p in ports:
        name = (p.device or "") + " " + (p.description or "")
        if any(k in name for k in ("Arduino", "wchusb", "ttyACM", "ttyUSB", "COM")):
            return p.device
    return ports[0].device if ports else None