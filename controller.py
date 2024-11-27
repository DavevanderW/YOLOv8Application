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
        self.model.addObserver(self.view)
        self.view.main()
    
    def _checkEmptyFields(self, YOLOv8ModelPathsList, imagesPath, outputPath):
        """
        Checks for empty fields. If an empty field is found, append to the errorMessage.

        Parameters:
            YOLOv8ModelPathsList (list[str]): List with the paths of the user's selected YOLOv8 Models. 
            imagesPath (str): Path of the user's selected folder with images where prediction is going to be performed on. 
            outputPath (str): Path of the user's selected folder where the output file(s) are being saved.

        Returns:
            errorMessage (str): The error message containing the empty fields.
        """

        errorMessage = ""
        # Check if all paths exist. When a file or folder does not exist, append the file or folder to the errormessage
        if not YOLOv8ModelPathsList:
            errorMessage = errorMessage + "Step 1: Please select at least one YOLOv8 model file.\n\n" 
        if not imagesPath:
            errorMessage = errorMessage + "Step 2: Please select an images folder.\n\n"
        if not outputPath:
            errorMessage = errorMessage + "Step 3: Please select an output folder."
        return errorMessage

    def _preparePredictionProgress(self, YOLOv8ModelPath, amountOfImages):
        """
        Prepares a part of the view to show the prediction progress for each model that is used in the prediction.

        Parameters:
            YOLOv8ModelPath (str): The path of the user's selected YOLOv8 Model that is currently used to perform prediction. 
            amountOfImages (int): The amount of images where prediction is going to be performed on. 
        """

        self.view.currentModel.set("Current model: " + YOLOv8ModelPath) # Set the name of the current model
        self.view.setPredictionProgressBarMaximum(amountOfImages) # Set the maximum of the ProgressBar to the amount of images for prediction

    def executePrediction(self, YOLOv8ModelPathsList, imagesPath, outputPath):
        """
        Checks all user's input and shows error if necessary. If all evrything is fine, prediction is executed for all selected models.

        Parameters:
            YOLOv8ModelPathsList (list[str]): List with the paths of the user's selected YOLOv8 Models. 
            imagesPath (str): Path of the user's selected folder with images where prediction is going to be performed on. 
            outputPath (str): Path of the user's selected folder where the output file(s) are being saved.
        """

        # Check if there are one or more empty fields that cannot be empty
        emptyFieldsErrorMessage = self._checkEmptyFields(YOLOv8ModelPathsList, imagesPath, outputPath)
        if emptyFieldsErrorMessage: 
            # One or more fields are empty, show an error indicating the empty fields
            self.view.showErrorMessageBox(emptyFieldsErrorMessage)
        else: 
            # All required fields are filled in, check if all paths exist
            notExistErrorMessage = self.model.pathsExist(YOLOv8ModelPathsList, imagesPath, outputPath)
            if notExistErrorMessage: 
                # One or more paths do not exist, show an error to the user containing the paths that don't exist
                self.view.showErrorMessageBox(notExistErrorMessage)
            else:
                # All paths exist, check if the images folder (imagesPath) actually contain any images and put the names of all images in a a list.
                imagesList = self.model.checkImagesFolder(imagesPath)
                if not imagesList: 
                    # If the images folder contains no images, show an error to the user.
                    self.view.showErrorMessageBox("Step 2: The chosen folder does not contain any images. Please choose a different folder with images.")
                else: 
                    # The images folder contains images, check if all YOLOv8 models are valid.
                    invalidYOLOv8ModelsList = self.model.validateYOLOv8Models(YOLOv8ModelPathsList)
                    if invalidYOLOv8ModelsList: 
                        # There are invalid YOLOv8 models, show an error to the user with the invalid models.
                        delimiter = "\n"
                        invalidModelsString = delimiter.join(invalidYOLOv8ModelsList)
                        self.view.showErrorMessageBox("The following YOLOv8 models are not valid YOLOv8 image classification or instance segmentation models: \n\n" + invalidModelsString)
                    else: 
                        # All models are valid, execute prediction with all selected models.
                        for YOLOv8ModelPath in YOLOv8ModelPathsList:
                            self._preparePredictionProgress(YOLOv8ModelPath, len(imagesList))
                            self.model.executeYOLOv8ModelPrediction(YOLOv8ModelPath, imagesPath, imagesList, outputPath)
                        self.view.showInfoMessageBox("Prediction completed. The results can be found in the selected output folder: \n\n" + outputPath)