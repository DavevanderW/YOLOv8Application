from model import Model
from view import View

class Controller():
    """ Class to act as a controller between the view and the model """
    
    def __init__(self):
        """ Initializes the Controller object """
        
        # Create a model
        self.model = Model()

        # Create a view, set the view as an observer for the model, and start the view
        self.view = View(self)
        self.model.add_observer(self.view)
        self.view.main()
    
    def _preparePredictionProgress(self, YOLOv8ModelPath, amountOfImages):
        """
        Prepares a part of the view to show the prediction progress for each model that is used in the prediction.

        Args:
            YOLOv8ModelPath (str): The path of the user's selected YOLOv8 Model that is currently used to perform prediction. 
            amountOfImages (int): The amount of images where prediction is going to be performed on. 
        """

        self.view.currentModel.set("Current model: " + YOLOv8ModelPath) # Set the name of the current model
        self.view.progress.set(0) # Reset the ProgressBar to zero
        self.view.setPredictionProgressBarMaximum(amountOfImages) # Set the maximum of the ProgressBar to the amount of images for prediction

    def executePrediction(self, YOLOv8ModelPathsList, imagesPath, outputPath):
        """
        Checks all user's input and shows error if necessary. If all evrything is fine, prediction is executed for all selected models.

        Args:
            YOLOv8ModelPathsList (list[str]): List with the paths of the user's selected YOLOv8 Models. 
            imagesPath (str): Path of the user's selected folder with images where prediction is going to be performed on. 
            outputPath (str): Path of the user's selected folder where the output file(s) are being saved.
        """

        # Check if there are one or more empty fields that cannot be empty
        if not YOLOv8ModelPathsList or not imagesPath or not outputPath: # If one or more fields are empty, show an error
            self.view.showErrorMessageBox("Some fields appear to be empty. Please choose at least one model in Step 1, the images folder in Step 2, and the output folder in Step 3 to predict.")
        else: 
            # All required fields are filled in, check if all paths exist
            errormessage = self.model.pathsExist(YOLOv8ModelPathsList, imagesPath, outputPath)
            if errormessage: 
                # One or more paths do not exist, show an error to the user containing the paths that don't exist
                self.view.showErrorMessageBox(errormessage)
            else:
                # All paths exist, check if the images folder (imagesPath) actually contain any images and put the names of all images in a a list.
                imagesList = self.model.checkImagesFolder(imagesPath)
                if not imagesList: 
                    # If the images folder contains no images, show an error to the user.
                    self.view.showErrorMessageBox("The chosen folder in Step 2 does not contain any images. Please choose a different folder with images.")
                else: 
                    # The images folder contains images, check if all YOLOv8 models are valid.
                    invalid_models = self.model.validateYOLOv8Models(YOLOv8ModelPathsList)
                    if invalid_models: 
                        # There are invalid YOLOv8 models, show an error to the user with the invalid models.
                        delimiter_invalid_models = "\n"
                        invalid_models_string = delimiter_invalid_models.join(invalid_models)
                        self.view.showErrorMessageBox("The following YOLOv8 models are not valid YOLOv8 image classification or instance segmentation models: \n\n" + invalid_models_string)
                    else: 
                        # All models are valid, execute prediction with all selected models.
                        for YOLOv8ModelPath in YOLOv8ModelPathsList:
                            self._preparePredictionProgress(YOLOv8ModelPath, len(imagesList))
                            self.model.executePrediction(YOLOv8ModelPath, imagesPath, imagesList, outputPath)
                        self.view.showInfoMessageBox("Prediction completed. The results can be found in the selected output folder: \n\n" + outputPath)