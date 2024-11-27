from pickle import UnpicklingError
from ultralytics import YOLO
import os
import csv

class Observable:
    def __init__(self):
        self.observers = []

    def addObserver(self, observer):
        if observer not in self.observers:
            self.observers.append(observer)

    def removeObserver(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def notifyObservers(self):
        for observer in self.observers:
            observer.update(self)

class Model(Observable):
    """ Class to interact with files, folders and YOLOv8 """

    def __init__(self):
        """ Initializes the Model object """
        super().__init__()
        self.progressCount = 0

    def _checkFileExists(self, filePathToCheck):
        """
        Checks if the path is an existing file.

        Parameters:
            fileToCheck (str): Path of the file.

        Returns:
            (bool): Boolean to indicate if the path is an existing file.
        """

        return os.path.isfile(filePathToCheck)

    def _checkFolderExists(self, folderPathToCheck):
        """
        Checks if the path is an existing folder.

        Parameters:
            folderPathToCheck (str): Path of the folder.

        Returns:
            (bool): Boolean to indicate if the path is an existing folder.
        """

        return os.path.isdir(folderPathToCheck)

    def _determineOutputFileName(self, outputPath):
        """
        Determines the output file name. The default output file name is "results.csv", but if a file with that name already exist try "results(1).csv", then "results(2).csv" and so on.
        
        Parameters:
            outputPath (str): Path of the user's selected folder where the output file(s) are being saved.

        Returns:
            outputFilePath (str): Path of the output file.
        """

        outputFilePath = os.path.join(outputPath, "results.csv")
        filename, extension = os.path.splitext(outputFilePath)
        counter = 1
        while os.path.exists(outputFilePath):
            outputFilePath = filename + " (" + str(counter) + ")" + extension
            counter += 1
        return outputFilePath

    def _getTaskOfYOLOv8Model(self, YOLOv8ModelToCheck):
        """
        Gets the task of a YOLOv8 Model.
        
        Parameters:
            YOLOv8ModelToCheck (str): Path of the YOLOv8 Model.

        Returns:
            (str): string that contains the task of the YOLOv8 Model.
        """
        
        model = YOLO(YOLOv8ModelToCheck)
        return model.task
    
    def _increaseProgressCount(self):
        """
        Increases the progressCount and notifies all observers
        """

        self.progressCount += 1
        self.notifyObservers()

    def _setProgressCount(self, progressCount):
        """
        Sets the progressCount and notifies all observers
        """

        self.progressCount = progressCount
        self.notifyObservers()
    
    def checkImagesFolder(self, imagesFolderPathToCheck):
        """
        Checks if the folder actually contains images (.png, .jpg, and .jpeg) and put the names of the images in a list.

        Parameters:
            imagesFolderPathToCheck (str): Path of the images folder.

        Returns:
            imagesList (list[str]): List that contains the names of all images in the images folder.
        """

        files = os.listdir(imagesFolderPathToCheck)
        imagesList = []
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                imagesList.append(file)
        return imagesList

    def executeYOLOv8ModelPrediction(self, YOLOv8ModelPath, imagesPath, imagesList, outputPath):
        """
        Executes a prediction with a single YOLOv8 Model on all images and puts the output in an output file.

        Parameters: 
            YOLOv8ModelPath (str): Path to the YOLOv8 Model that is used for prediction.
            imagesPath (str): Path of the user's selected folder with images where prediction is going to be performed on.
            imagesList (list[str]): List that contains the names of all images in the images folder.
            outputPath (str): Path of the user's selected folder where the output file(s) are being saved.
        """
        self._setProgressCount(0) # Reset the progress count to zero
        task = self._getTaskOfYOLOv8Model(YOLOv8ModelPath)
        if task == "classify":
            YOLOv8Model = YOLO(YOLOv8ModelPath) # Load the YOLOv8 model you want to predict with

            outputFilePath = self._determineOutputFileName(outputPath) # determine the output file name

            with open(outputFilePath, "w", newline='') as output: # create the csv file
                outputWriter = csv.writer(output) # create the csv writer
                outputWriter.writerow(["image_name", "predicted_class", "confidence"]) # write the first row

                # Loop through all images (.png, .jpg, .jpeg) and perform prediction on them, save each prediction in the csv file
                for image in imagesList:
                    results = YOLOv8Model(os.path.join(imagesPath, image)) # Perform prediction on the image
                    predictedClass = results[0].names[results[0].probs.top1] # Get the name of the class with the highest confidence
                    predictionConfidence = results[0].probs.top1conf.cpu().item() # Get the confidence of the predicted class
                    outputWriter.writerow([image, predictedClass, predictionConfidence]) # Write the prediction results
                    self._increaseProgressCount() # Prediction has been performed on this image, increase the progress count
            del YOLOv8Model

        elif task == "segment":
            YOLOv8Model = YOLO(YOLOv8ModelPath) # Load the YOLOv8 model you want to predict with

            outputFilePath = self._determineOutputFileName(outputPath) # determine the output file name

            with open(outputFilePath, "w", newline='') as output: # create the csv file
                outputWriter = csv.writer(output) # create the csv writer
                outputWriter.writerow(["image_name", "amount_of_instances"]) # write the first row

                # Loop through all images (.png, .jpg, .jpeg) and perform prediction on them, save each prediction in the csv file
                for image in imagesList:
                    results = YOLOv8Model(os.path.join(imagesPath, image)) # Perform prediction on the image
                    if results[0].masks != None:
                        amountOfInstances = len(results[0].masks.xyn) # Get the amount of detected masks
                    else: 
                        amountOfInstances = 0 # There are zero detections
                    outputWriter.writerow([image, amountOfInstances]) # Write the prediction results
                    self._increaseProgressCount() # Prediction has been performed on this image, increase the progress count
            del YOLOv8Model

    def pathsExist(self, YOLOv8ModelPathsList, imagesPath, outputPath):
        """ 
        Checks if all paths in the arguments are existing paths.
        
        Parameters:
            YOLOv8ModelPathsList (list[str]): List with the paths of the user's selected YOLOv8 Models. 
            imagesPath (str): Path of the user's selected folder with images where prediction is going to be performed on. 
            outputPath (str): Path of the user's selected folder where the output file(s) are being saved.

        Returns: 
            errormessage (str): The error message containing the paths of files and/or folders that don't exist.
        """

        errorMessage = ""
        # Check if all paths exist. When a file or folder does not exist, append the file or folder to the errormessage
        for YOLOv8ModelPath in YOLOv8ModelPathsList:
            if not self._checkFileExists(YOLOv8ModelPath):
                errorMessage = errorMessage + "Step 1: YOLOv8 model file not found:\n"  + YOLOv8ModelPath + "\n\n" 
        if not self._checkFolderExists(imagesPath):
            errorMessage = errorMessage + "Step 2: Images folder not found:\n" + imagesPath + "\n\n"
        if not self._checkFolderExists(outputPath):
             errorMessage = errorMessage + "Step 3: Output folder not found:\n" + outputPath + "\n\n"
        return errorMessage

    def validateYOLOv8Models(self, YOLOv8ModelPathsList):
        """
        Checks if all models are valid YOLOv8 Models. Loops through YOLOv8ModelPathsList, if an UnpicklingError is catched, add the path to the invalidYOLOv8ModelsPathsList

        Parameters:
            YOLOv8ModelPathsList (list[str]): List with the paths of the user's selected YOLOv8 Models. 

        Returns:
            invalidYOLOv8ModelPathsList (list[str]): List with the paths of unvalid YOLOv8 Models.
        """

        invalidYOLOv8ModelPathsList = []
        for YOLOv8ModelPath in YOLOv8ModelPathsList:
            try:
                self._getTaskOfYOLOv8Model(YOLOv8ModelPath)
            except UnpicklingError:
                invalidYOLOv8ModelPathsList.append(YOLOv8ModelPath)
        return invalidYOLOv8ModelPathsList