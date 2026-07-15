import cv2

class ImageProcessor:

    def __init__(self, detector):
        self.detector = detector

    def process(self, image):

        detections = self.detector.detect(image)

        colors = {
            "person": (0,255,0),
            "helmet": (255,0,0),
            "safety_vest": (0,255,255),
            "truck": (255,255,0),
            "forklift": (255,0,255),
            "excavator": (0,128,255),
            "fire": (0,0,255),
            "smoke": (128,128,128)
        }

        for d in detections:

            x1,y1,x2,y2 = d["bbox"]

            class_name = d["class_name"]

            conf = d["confidence"]

            color = colors.get(class_name,(0,255,0))

            cv2.rectangle(
                image,
                (x1,y1),
                (x2,y2),
                color,
                2
            )

            label = f"{class_name} {conf:.2f}"

            cv2.putText(
                image,
                label,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return image, detections