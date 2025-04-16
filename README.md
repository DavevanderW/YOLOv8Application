# YOLOv8 Application
A simple Python application for applying between one and five YOLOv8 models on a folder of pictures.

## Getting started
- Clone this repository
- Install all dependencies using `pip3 install -r requirements.txt`

## Using the application
- Start the application by running YOLOv8Application.py
- Step 1: Select between one and five YOLOv8 models
- Step 2: Select the folder with the images you want to apply the YOLOv8 models on
- Step 3: Select a folder where the output files will be saved to
- Press the 'Predict!' button to start the application of the chosen YOLOv8 models on the images
- During the prediction, the progress bar shows the current predicting YOLOv8 model and his progress on the images

## Notes
- The application filters the files in the chosen images folder on the image files extensions .png, .jpg and .jpeg. These extensions can be modified in method 'checkImagesFolder' in model.py.
- The application currently only supports YOLOv8 image classification and instance segmentation models. Support for other types of YOLOv8 models can be added, by adding the new type to the dict variable 'supportedYOLOv8ModelTasks' in model.py. Next, method 'executeYOLOv8ModelPrediction' in model.py must be modified to support the execution of the new model type.
- The output for different types of YOLOv8 models can be modified, by modifying the method 'executeYOLOv8ModelPrediction' in model.py.
  - Currently for image classfication the iamge name, the name of the predicted class (with the highest confidence) and the confidence score is saved in the csv file (one line per image).
  - Currently for instance segmentation only the iamge name and the number of detected masks is saved in the csv file (one line per image).
 
## License
Distributed under the MIT License. See `LICENSE` for more information.
