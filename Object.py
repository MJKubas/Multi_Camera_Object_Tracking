import numpy as np
import cv2

class ObjectTracker:
    def __init__(self, color, bbox, classLabel, id):
        self.color = color
        self.id = id
        self.bbox = bbox
        self.classLabel = classLabel
        self.leftToSaveDesc = 30
        self.isNew = True
        self.leftToDelete = 30

    def drawBoxes(self, frame):
        color = self.color
        drawbox = self.bbox
        cv2.rectangle(frame, drawbox, color, 2)
        cv2.rectangle(frame, (drawbox[0], drawbox[1] - 20), (drawbox[0] + drawbox[2], drawbox[1]), color, -1)
        cv2.putText(frame, str(self.id), (drawbox[0], drawbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
        return frame


class ObjectReID:
    def __init__(self, classLabel, color, description):
        if color == None:
            self.color = ObjectReID.random_colour()
        else:
            self.color = color
        self.id = int(str(self.color[0] + self.color[1] + self.color[2]))
        self.classLabel = classLabel
        self.description = []
        if len(description[0]) != 0:
            self.description.append(description)
        self.newObjectCounter = 60

    def random_colour():
        color1 = (list(np.random.choice(range(256), size=3)))  
        color =[int(color1[0]), int(color1[1]), int(color1[2])]  
        return color