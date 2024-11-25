from pickle import UnpicklingError
from ultralytics import YOLO
import os
import csv

class Observable:
    def __init__(self):
        self.observers = []

    def add_observer(self, observer):
        if observer not in self.observers:
            self.observers.append(observer)

    def remove_observer(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def notify_observers(self):
        for observer in self.observers:
            observer.update(self)

class Model(Observable):
    def __init__(self):
        super().__init__()
        self.progressCount = 0

    def getTaskOfYOLOv8Model(self, YOLOv8modelToCheck):
        model = YOLO(YOLOv8modelToCheck)
        return model.task
    
    def checkFileExists(self, fileToCheck):
        return os.path.isfile(fileToCheck)
    
    def checkFolderExists(self, folderPathToCheck):
        return os.path.isdir(folderPathToCheck)
    
    def checkImagesFolder(self, imagesFolderPathToCheck):
        files = os.listdir(imagesFolderPathToCheck)
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                return True
        return False

    def _determineOutputFileName(self, outputPath):
        # default output file is "results.csv", but if a file with that name already exist try "results(1).csv", then "results(2).csv" and so on.
        outputFilePath = os.path.join(outputPath, "results.csv")
        filename, extension = os.path.splitext(outputFilePath)
        counter = 1

        while os.path.exists(outputFilePath):
            outputFilePath = filename + " (" + str(counter) + ")" + extension
            counter += 1

        return outputFilePath

    def validateYOLOv8Models(self, YOLOv8ModelPathsList):
        invalidYOLOv8ModelPathsList = []
        
        # Loop through the YOLOv8Models. If an UnpicklingError is catched, add the model to the invalidYOLOv8ModelsPathsList
        for YOLOv8ModelPath in YOLOv8ModelPathsList:
            try:
                self.getTaskOfYOLOv8Model(YOLOv8ModelPath)
            except UnpicklingError:
                invalidYOLOv8ModelPathsList.append(YOLOv8ModelPath)
        return invalidYOLOv8ModelPathsList

    def increaseProgressCount(self):
        self.progressCount += 1
        self.notify_observers()

    def executePrediction(self, YOLOv8ModelPathsList, imagesPath, outputPath):
        for YOLOv8ModelPath in YOLOv8ModelPathsList:
            task = self.getTaskOfYOLOv8Model(YOLOv8ModelPath)
            self.progressCount = 0

            if task == "classify":
                YOLOv8Model = YOLO(YOLOv8ModelPath) # Load the model you want to predict with

                outputFilePath = self._determineOutputFileName(outputPath) # determine the output file name

                with open(outputFilePath, "w", newline='') as output: # create the csv file
                    output_writer = csv.writer(output) # create the csv writer
                    output_writer.writerow(["image_name", "predicted_class", "confidence"]) # write the first row

                    # Loop through all images (.png, .jpg, .jpeg) in the folder and perform prediction on them, save each prediction in the csv file
                    files = os.listdir(imagesPath)
                    for file in files:
                        if file.endswith((".png", ".jpg", ".jpeg")):
                            results = YOLOv8Model(os.path.join(imagesPath, file)) # Perform prediction on the image
                            predicted_class = results[0].names[results[0].probs.top1] # Get the name of the class with the highest confidence
                            prediction_confidence = results[0].probs.top1conf.cpu().item() # Get the confidence of the predicted class
                            output_writer.writerow([file, predicted_class, prediction_confidence]) # Write the prediction results
                            self.increaseProgressCount()
                del YOLOv8Model

            elif task == "segment":
                YOLOv8Model = YOLO(YOLOv8ModelPath) # Load the model you want to predict with

                outputFilePath = self._determineOutputFileName(outputPath) # determine the output file name

                with open(outputFilePath, "w", newline='') as output: # create the csv file
                    output_writer = csv.writer(output) # create the csv writer
                    output_writer.writerow(["image_name", "amount_of_instances"]) # write the first row

                    # Loop through all images (.png and .jpg) in the folder and perform prediction on them, save each prediction in the csv file
                    files = os.listdir(imagesPath)
                    for file in files:
                        if file.endswith((".png", ".jpg", ".jpeg")):
                            results = YOLOv8Model(os.path.join(imagesPath, file)) # Perform prediction on the image
                            if results[0].masks != None:
                                amount_of_instances = len(results[0].masks.xyn) # Get the amount of detected masks
                            else: 
                                amount_of_instances = 0
                            output_writer.writerow([file, amount_of_instances]) # Write the prediction results
                            self.increaseProgressCount()
                del YOLOv8Model