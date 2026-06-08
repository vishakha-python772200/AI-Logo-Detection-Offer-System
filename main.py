import os
import shutil
import random # jar suffel nahi kela tr train and validation madhe bias yeu shakt 
import cv2
from ultralytics import YOLO # for classifaication the model

# PATHS

BASE_PATH    = r"E:\VISHAKHA PYTHON ALL IMPORTANT DATA VERY IMPORTANT DATA\Vishakha ml small project\computer_vision"
LOGOS_PATH   = os.path.join(BASE_PATH, "logos") # raw images astil
DATASET_PATH = os.path.join(BASE_PATH, "dataset")# for yolo training 
RUNS_PATH    = os.path.join(BASE_PATH, "runs") # yolo training cha data saved hoto itha 
MODEL_PATH   = os.path.join(RUNS_PATH, "classify", "train", "weights", "best.pt") # final train model sathi 

# ==========================================
# 10 BRANDS - OFFER + COLOR
# ==========================================
ALL_BRANDS = {
    "kfc":      {"offer": "50% OFF ON ALL COMBOS TODAY",      "color": (0, 0, 200)},
    "mcd":      {"offer": "BUY 1 BURGER GET 1 FREE",          "color": (0, 165, 255)},
    "vw":       {"offer": "1 DAY FREE TEST DRIVE",            "color": (200, 0, 0)},
    "bata":     {"offer": "50% OFF ON ALL FOOTWEAR",          "color": (0, 100, 200)},
    "bmw":      {"offer": "NEW MODEL LAUNCH - BOOK NOW",      "color": (50, 50, 50)},
    "dominos":  {"offer": "BUY 1 PIZZA GET 1 FREE",           "color": (0, 0, 180)},
    "fastrack": {"offer": "20% OFF ON ALL WATCHES",           "color": (180, 0, 180)},
    "haldiram": {"offer": "BUY 1 GET 1 FREE ON SWEETS",       "color": (0, 140, 255)},
    "starbuks": {"offer": "BUY 2 COFFEE GET 1 FREE",          "color": (0, 100, 0)},
    "zudio":    {"offer": "10% OFF ON ALL CLOTHES",           "color": (130, 0, 130)},
}

# ==========================================
# SIRF JE FOLDERS EXIST KARTO TE BRANDS
# ==========================================
BRANDS = []
for brand in ALL_BRANDS:
    folder = os.path.join(LOGOS_PATH, brand)
    if os.path.exists(folder):
        BRANDS.append(brand)

print(f"Found brands: {[b.upper() for b in BRANDS]}\n")

# ==========================================
# STEP 1: FOLDERS BANAV
# ==========================================
print("=" * 50)
print("STEP 1 : Folders Banvtoy...")
print("=" * 50)

for brand in BRANDS:
    os.makedirs(os.path.join(DATASET_PATH, "train", brand), exist_ok=True)
    os.makedirs(os.path.join(DATASET_PATH, "val",   brand), exist_ok=True)
    print(f"Ready : {brand}")

print("Folders Done!\n")

# ==========================================
# STEP 2: IMAGES SPLIT 80% TRAIN 20% VAL
# ==========================================
print("=" * 50)
print("STEP 2 : Images Split Kartoy...")
print("=" * 50)

for brand in BRANDS:

    src = os.path.join(LOGOS_PATH, brand)

    all_images = [
        f for f in os.listdir(src)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    random.shuffle(all_images)

    split      = int(len(all_images) * 0.8)
    train_imgs = all_images[:split]
    val_imgs   = all_images[split:]

    for img in train_imgs:
        dst = os.path.join(DATASET_PATH, "train", brand, img)
        if not os.path.exists(dst):
            shutil.copy(os.path.join(src, img), dst)

    for img in val_imgs:
        dst = os.path.join(DATASET_PATH, "val", brand, img)
        if not os.path.exists(dst):
            shutil.copy(os.path.join(src, img), dst)

    print(f"{brand.upper():12} : {len(train_imgs)} train | {len(val_imgs)} val")

print("Split Done!\n")

# ==========================================
# STEP 3: YOLO TRAINING
# ==========================================
print("=" * 50)
print("STEP 3 : YOLOv8 Training Suru!")
print("=" * 50)
print("30-45 minutes lagtil...")
print("Laptop band karu nakos!\n")

model = YOLO("yolov8n-cls.pt")

model.train(
    data    = DATASET_PATH,
    epochs  = 100,
    imgsz   = 224,
    batch   = 16,
    device  = "cpu",
    workers = 2,
    patience= 20,
    lr0     = 0.001,
    save    = True,
    plots   = True,
    verbose = True,
    project = RUNS_PATH,
    name    = "classify/train"
)

print("\nTraining Zali!")
print(f"Model : {MODEL_PATH}\n")

# ==========================================
# STEP 4: MODEL LOAD
# ==========================================
print("=" * 50)
print("STEP 4 : Testing Suru!")
print("=" * 50)

if not os.path.exists(MODEL_PATH):
    print(f"Model nahi milala : {MODEL_PATH}")
    exit()

best_model = YOLO(MODEL_PATH)
print("Model Load Zala!\n")

# ==========================================
# COUNTERS
# ==========================================
total         = 0
correct       = 0
brand_total   = {b: 0 for b in BRANDS}
brand_correct = {b: 0 for b in BRANDS}

# ==========================================
# STEP 5: TESTING LOOP
# ==========================================
for cls in BRANDS:

    folder = os.path.join(LOGOS_PATH, cls)

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    print("=" * 55)
    print(f"TESTING : {cls.upper()} ({len(images)} images)")
    print("=" * 55)

    for img_name in images:

        img_path = os.path.join(folder, img_name)
        frame    = cv2.imread(img_path)

        if frame is None:
            continue

        frame = cv2.resize(frame, (800, 500))

        # ==========================================
        # YOLO PREDICTION
        # ==========================================
        results    = best_model.predict(
                        source  = frame,
                        imgsz   = 224,
                        verbose = False
                    )
        r          = results[0]

        if r.probs is None:
            continue

        pred_id    = int(r.probs.top1)
        pred_name  = r.names[pred_id].lower()
        confidence = r.probs.top1conf.item() * 100

        brand_total[cls] += 1
        total += 1

        # ==========================================
        # CORRECT / WRONG
        # ==========================================
        is_correct = (pred_name == cls.lower())

        if is_correct:
            brand_correct[cls] += 1
            correct += 1

        # ==========================================
        # BRAND INFO
        # ==========================================
        brand_info   = ALL_BRANDS.get(pred_name, {
            "offer": "SPECIAL OFFER AVAILABLE",
            "color": (80, 80, 80)
        })
        banner_color = brand_info["color"]
        offer_text   = brand_info["offer"]

        # ==========================================
        # UI - TOP BANNER - Brand + Confidence
        # ==========================================
        cv2.rectangle(frame, (0, 0), (800, 65), banner_color, -1)
        cv2.putText(frame,
                    f"Detected: {pred_name.upper()}  |  Confidence: {confidence:.1f}%",
                    (15, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (255, 255, 255), 2)

        # ==========================================
        # UI - MIDDLE BANNER - Offer
        # ==========================================
        cv2.rectangle(frame, (0, 65), (800, 125), (20, 20, 20), -1)
        cv2.putText(frame,
                    f"OFFER : {offer_text}",
                    (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0, 255, 255), 2)

        # ==========================================
        # UI - BOTTOM - Result
        # ==========================================
        if is_correct:
            result_color = (0, 255, 0)
            result_msg   = f"CORRECT - Expected: {cls.upper()}"
        else:
            result_color = (0, 0, 255)
            result_msg   = f"WRONG - Expected: {cls.upper()} | Got: {pred_name.upper()}"

        cv2.rectangle(frame, (0, 455), (800, 500), (15, 15, 15), -1)
        cv2.putText(frame,
                    result_msg,
                    (15, 485),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, result_color, 2)

        print(result_msg)

        cv2.imshow("AI Logo Detection", frame)

        key = cv2.waitKey(400)
        if key == 27:
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()

# ==========================================
# FINAL REPORT
# ==========================================
print("\n")
print("=" * 60)
print("        AI LOGO DETECTION - FINAL REPORT")
print("=" * 60)

for brand in BRANDS:
    if brand_total[brand] > 0:
        acc    = (brand_correct[brand] / brand_total[brand]) * 100
        status = "EXCELLENT" if acc >= 90 else "GOOD" if acc >= 75 else "NEEDS MORE DATA"
        print(f"{brand.upper():12} | {brand_correct[brand]:3}/{brand_total[brand]:3} | {acc:6.2f}% | {status}")

print("=" * 60)

if total > 0:
    overall = (correct / total) * 100
    print(f"{'OVERALL':12} | {correct:3}/{total:3} | {overall:.2f}%")

    if overall >= 90:
        print("STATUS : EXCELLENT - Project Complete!")
    elif overall >= 75:
        print("STATUS : GOOD - Thoda improve karu!")
    else:
        print("STATUS : Jast images hav yat!")

print("=" * 60)