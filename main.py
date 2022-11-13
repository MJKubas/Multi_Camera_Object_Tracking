from Tracker import *

def camera(videoPath, objectNS, listOfCameras):
    detector = Detector(videoPath, objectNS, listOfCameras)
    detector.startTracking()

def main():
    processes = []
    videoPaths = ["C:\Magisterka\Korytarz2.mp4", "C:\Magisterka\Sypialnia2.mp4"] 
    mgr = multiprocessing.Manager()
    objectsNS = mgr.Namespace()

    for cam in videoPaths:
        externalVideoPaths = videoPaths.copy()
        externalVideoPaths.remove(cam)
        d = multiprocessing.Process(target=camera, args=(cam, objectsNS, externalVideoPaths, ))
        objectsNS.__setattr__(cam, [])
        processes.append(d)

    for d in processes:
        d.start()
        
    for d in processes:
        d.join()


if __name__ == "__main__":
    main()