import cv2
import numpy as np
import multiprocessing
import time
from Object import *
from ReID import ReID

INTERSECT = 50
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

class Detector:
    detectedObjects = []

    net = cv2.dnn.readNet("C:/Magisterka/model/yolov6s.onnx")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def __init__(self, videoPath, objectNS, listOfCameras):
        self.listOfCameras = listOfCameras
        self.objectNS = objectNS
        self.videoPath = videoPath
        self.objects = []
        self.oldObjects = []

    def detect(self, image, net):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        preds = net.forward()
        return preds

    #process detected entities
    def wrapDetection(self, input_image, output_data):
        class_ids = []
        confidences = []
        boxes = []

        rows = output_data.shape[0]

        image_width, image_height, _ = input_image.shape

        x_factor = image_width / INPUT_WIDTH
        y_factor =  image_height / INPUT_HEIGHT

        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= 0.4: #get only those with confidence >= 0.4

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                
                if (class_id == 0 or class_id == 15 or class_id == 16) and classes_scores[class_id] > .25: #Get only person, cat or dog class
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

        result = []

        for i in indexes:
            result.append([class_ids[i], boxes[i]])
        return result

    #transform image to yolo format
    def formatYolo(self, frame):
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    def getFrame(ns, event, ipAddress):
        capture = cv2.VideoCapture(ipAddress)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        while True:
            if capture.isOpened():
                start = time.time()
                (status, frame) = capture.read()
                if not status:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame = cv2.resize(frame, [960, 540])
                ns.value = frame
                event.set()

                stop = time.time()
                t = stop - start
                if t < (1/10):
                    time.sleep((1/10)-t)

    # calculate intersection of two boxes
    def checkIntersection(self, boxA, boxB):
        if type(boxA) == type(None) or type(boxB) == type(None):
            return 0

        area1 = boxA[2]*boxA[3]
        area2 = boxB[2]*boxB[3]
            
        x = max(boxA[0], boxB[0])
        y = max(boxA[1], boxB[1])
        w = min(boxA[0] + boxA[2], boxB[0] + boxB[2]) - x
        h = min(boxA[1] + boxA[3], boxB[1] + boxB[3]) - y

        if w < 0 or h < 0:
            return 0

        else:
            intersectArea = w*h
            if area1 >= area2:
                return (intersectArea/area2)*100
            else:
                return (intersectArea/area1)*100

    def updateDetection(self, ns, event, ns2, event2):
        while True:
            try:
                frame = ns.value
            except Exception as err:
                print('No frame: ', str(err))
            event.wait()
            detected = []

            inputImage = self.formatYolo(ns.value)
            outs = self.detect(inputImage, self.net)
            detected = self.wrapDetection(inputImage, outs[0])

            ns2.value = detected
            event2.set()

    def startTracking(self):
        reID = ReID()

        mgr = multiprocessing.Manager()
        namespace = mgr.Namespace()
        frameEvent = multiprocessing.Event()
        mgr2 = multiprocessing.Manager()
        namespace2 = mgr2.Namespace()
        boxesEvent = multiprocessing.Event()
        camera = multiprocessing.Process(target=Detector.getFrame, args=(namespace, frameEvent, self.videoPath))
        detection = multiprocessing.Process(target=self.updateDetection, args=(namespace, frameEvent, namespace2, boxesEvent))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        backSub = cv2.bgsegm.createBackgroundSubtractorGMG()
        camera.daemon = True
        camera.start()
        detection.daemon = True
        detection.start()

        try:
            image = namespace.value
        except Exception as err:
            print('No frame: ', str(err))
        frameEvent.wait()
        image = namespace.value

        try:
            detected = namespace2.value
        except Exception as err:
            print('No detection: ', str(err))
        boxesEvent.wait()
        detected = namespace2.value

        startTime = 0

        while True:
            currentTime = time.time()
            fps = 1/(currentTime-startTime)
            startTime = currentTime

            self.objectNS.__setattr__(self.videoPath, reID.GetDetectedObjects())

            externalObjects = []
            for cam in self.listOfCameras:
                externalObjects += self.objectNS.__getattr__(cam)

            reID.SetExternalDetectedObjects(externalObjects)

            #background substraction
            image_copy = image.copy()
            fgMask = backSub.apply(image_copy)

            fgMask = cv2.medianBlur(fgMask, 9)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)

            for detect in detected:
                intersectValue = INTERSECT
                index = -1
                x,y,w,h = detect[1]
                ROI = image_copy[y:y+h, x:x+w]
                ROIMask = fgMask[y:y+h, x:x+w]
                
                try:
                    ROI = cv2.bitwise_and(ROI, ROI, mask=ROIMask)
                    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                except:
                    continue
                
                for i in range(len(self.oldObjects)):
                    if detect[0] == self.oldObjects[i].classLabel:
                        inter = self.checkIntersection(detect[1], self.oldObjects[i].bbox)
                        if inter > intersectValue:
                            intersectValue = inter
                            index =  i
                            self.oldObjects[i].leftToDelete = 30

                if index != -1:
                    self.oldObjects[index].bbox = detect[1]
                    if self.oldObjects[index].isNew:
                        newId, color = reID.CheckNewObject(ROI, self.oldObjects[index].id, self.oldObjects[index].classLabel)
                        if newId == -2:
                            self.oldObjects[index].isNew = False
                        elif newId != -1:
                            self.oldObjects[index].color = color
                            self.oldObjects[index].id = newId
                            self.oldObjects[index].isNew = False
                        self.oldObjects[index].leftToSaveDesc = 30

                    elif self.oldObjects[index].leftToSaveDesc <= 0:
                        if reID.UpdateObjectDescriptors(ROI, self.oldObjects[index].id):
                            self.oldObjects[index].leftToSaveDesc = 30
                    self.oldObjects[index].leftToSaveDesc -= 1

                    self.objects.append(self.oldObjects[index])
                    self.oldObjects.pop(index)

                if index == -1:
                    detectedObject, isNew = reID.ObjectReID(ROI, detect[0])
                    if type(detectedObject) != type(None):
                        objectToSave = ObjectTracker(detectedObject.color, detect[1], detectedObject.classLabel, detectedObject.id)
                        if not isNew:
                            objectToSave.isNew = False
                        self.objects.append(objectToSave)

            for object in self.objects:
                if not object.isNew:
                    image = object.drawBoxes(image)

            for oldObject in self.oldObjects: #check if should be deleted when not detected
                if oldObject.leftToDelete > 0:
                    oldObject.leftToDelete -= 1
                    self.objects.append(oldObject)

            self.oldObjects = self.objects.copy()
            self.objects = []
            
                    
            cv2.putText(image, "FPS: " + str(int(fps)), (20,70), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
            image = cv2.resize(image, [960, 540])
            cv2.imshow(self.videoPath, image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            image = namespace.value
            detected = namespace2.value

        cv2.waitKey(0)
        cv2.destroyAllWindows()