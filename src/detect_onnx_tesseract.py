import argparse, time, re, cv2, numpy as np
import onnxruntime as ort
import pytesseract

# Simple ONNX inference for NanoDet-like outputs.
# Assumes output is (batch, num_anchors, num_classes+reg) typical NanoDet ONNX.
# For simplicity, we expect a model exported with NanoDet default postprocess disabled;
# we'll implement a minimal decoder for 1-class using score thresholding.

def letterbox(img, new_shape=(320,320), color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(new_shape[0]/h, new_shape[1]/w)
    new_unpad = (int(round(w*r)), int(round(h*r)))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw /= 2; dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, int(round(dh)), int(round(dh)), int(round(dw)), int(round(dw)),
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def send_udp(host, port, msg):
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(msg.encode("utf-8"), (host, port))

def send_serial(port, baud, msg):
    import serial
    ser = serial.Serial(port, baudrate=baud, timeout=1)
    ser.write((msg+"\n").encode("utf-8"))
    ser.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--min-score", type=float, default=0.35)
    ap.add_argument("--max-det", type=int, default=3)
    ap.add_argument("--pattern", default=r"(\\d{3}\\s?[A-Z]\\d{2})")
    ap.add_argument("--send", choices=["none","udp","serial"], default="none")
    ap.add_argument("--udp-host", default="192.168.0.50")
    ap.add_argument("--udp-port", type=int, default=5005)
    ap.add_argument("--serial-port", default="COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--all", action="store_true", help="OCR all boxes instead of top-1")
    args = ap.parse_args()

    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(args.model, providers=providers)
    iname = sess.get_inputs()[0].name
    onames = [o.name for o in sess.get_outputs()]

    cap = cv2.VideoCapture(args.camera)
    last_sent = ""
    stable_text = ""
    stable_since = 0

    pat = re.compile(args.pattern)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        img0 = frame.copy()
        img, r, (dw, dh) = letterbox(img0, (args.imgsz, args.imgsz))
        blob = img.astype(np.float32) / 255.0
        blob = blob.transpose(2,0,1)[None, ...]

        out = sess.run(onames, {iname: blob})[0]

        # Heuristic postprocess: assume out shape [1, N, 5] -> [score, cx, cy, w, h] already decoded (depends on export).
        # If not, you'll need to adapt to your exported model's format.
        # Here we handle either [1,N,6] with [x1,y1,x2,y2,score,cls] or [1,N,5] with [score,cx,cy,w,h].
        dets = out.squeeze()
        boxes = []
        if dets.ndim == 2 and dets.shape[1] >= 5:
            for d in dets:
                if dets.shape[1] >= 6:
                    x1,y1,x2,y2,score,cls = d[:6]
                else:
                    score,cx,cy,w,h = d[:5]
                    if score < args.min-score:
                        continue
                    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
                score = float(d[-2]) if dets.shape[1] >= 6 else float(score)
                if score < args.min_score:
                    continue
                # scale back to original image
                x1 = (x1 - dw) / r
                y1 = (y1 - dh) / r
                x2 = (x2 - dw) / r
                y2 = (y2 - dh) / r
                x1,y1,x2,y2 = map(lambda v:int(max(0,v)), [x1,y1,x2,y2])
                boxes.append((score, x1,y1,x2,y2))

        boxes = sorted(boxes, key=lambda x: -x[0])[:args.max_det]

        ocr_texts = []
        crops = []
        for i,(score,x1,y1,x2,y2) in enumerate(boxes):
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            roi = img0[y1:y2, x1:x2]
            if roi.size == 0: continue
            # Simple preproc for OCR
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(gray, lang="eng")
            m = pat.findall(text)
            if m:
                ocr_texts.extend(m)
            crops.append(roi)

            if not args.all and ocr_texts:
                break

        display_txt = ", ".join(ocr_texts) if ocr_texts else "(no match)"
        cv2.putText(frame, display_txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Debounce + send once per stable reading
        if ocr_texts:
            if stable_text != ocr_texts[0]:
                stable_text = ocr_texts[0]
                stable_since = time.time()
            elif time.time()-stable_since > 0.5 and last_sent != stable_text:
                # send
                if args.send == "udp":
                    send_udp(args.udp_host, args.udp_port, stable_text)
                elif args.send == "serial":
                    send_serial(args.serial_port, args.baud, stable_text)
                last_sent = stable_text
                print("SENT:", stable_text)

        cv2.imshow("Waybill Detect + OCR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()