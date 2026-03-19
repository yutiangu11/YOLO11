from ultralytics import YOLO

model = YOLO(r"yolo11n.pt")

model.predict(
    source = r"E:\YOLO\ultralytics-main\ultralytics-main\runs\detect\video\111.mp4",
    save = True,
    show = True
)
