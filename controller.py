from model import Model
from view import View

class Controller():
    def __init__(self):
        
        #create a model
        self.model = Model()

        # create a view and start the view
        self.view = View(self)
        self.model.add_observer(self.view)
        self.view.main()

    def _pathsExist(self, YOLOv8ModelPathsList, imagesPath, outputPath):
        errormessage = ""
        # Check if all paths exist, if a file or folder does not exis, append the error to the errormessage
        for YOLOv8ModelPath in YOLOv8ModelPathsList:
            if not self.model.checkFileExists(YOLOv8ModelPath):
                errormessage = errormessage + "YOLOv8 Model File not found:\n"  + YOLOv8ModelPath + "\n\n" 
        if not self.model.checkFolderExists(imagesPath):
            errormessage = errormessage + "Images Folder not found:\n" + imagesPath + "\n\n"
        if not self.model.checkFolderExists(outputPath):
             errormessage = errormessage + "Output folder not found:\n" + outputPath + "\n\n"
        return errormessage
    
    def _preparePredictionProgress(self, YOLOv8ModelPath, imagesList):
        self.view.currentModel.set("Current model: " + YOLOv8ModelPath)
        self.view.progress.set(0) # Reset the ProgressBar to zero
        maximumProgress = len(imagesList)
        self.view.setPredictionProgressBarMaximum(maximumProgress) # Set the maximum of the ProgressBar to the amount of images for prediction

    def executePrediction(self, YOLOv8ModelPathsList, imagesPath, outputPath):
        if not YOLOv8ModelPathsList or not imagesPath or not outputPath: # If one or more fields are empty, show an error
            self.view.showErrorMessageBox("Some fields appear to be empty. Please choose at least one model in Step 1, the images folder in Step 2, and the output folder in Step 3 to predict.")
        else:
            errormessage = self._pathsExist(YOLOv8ModelPathsList, imagesPath, outputPath) # Check if all paths exist
            imagesList = self.model.checkImagesFolder(imagesPath) # Check if the images folder contains images, if so add theire paths to the list
            if errormessage: # One or more paths do not exist, show an error to the user
                self.view.showErrorMessageBox(errormessage)
            elif not imagesList: # If the images folder contains no images, show an error to the user
                self.view.showErrorMessageBox("The chosen folder in Step 2 does not contain any images. Please choose a different folder with images.")
            else: 
                # All paths exist and the images folder contains images, check if all YOLOv8 models are valid
                invalid_models = self.model.validateYOLOv8Models(YOLOv8ModelPathsList)
                
                if invalid_models: # There are invalid YOLOv8 models, show an error to the user
                    delimiter_invalid_models = "\n"
                    invalid_models_string = delimiter_invalid_models.join(invalid_models)
                    self.view.showErrorMessageBox("The following YOLOv8 models are not valid YOLOv8 image classification or instance segmentation models: \n\n" + invalid_models_string)
                else: # All models are valid, execute prediction
                    for YOLOv8ModelPath in YOLOv8ModelPathsList:
                        self._preparePredictionProgress(YOLOv8ModelPath, imagesList)
                        self.model.executePrediction(YOLOv8ModelPath, imagesPath, imagesList, outputPath)
                    self.view.showInfoMessageBox("Prediction completed. The results can be found in the selected output folder: \n\n" + outputPath)