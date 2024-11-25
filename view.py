import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from model import Model

class Observer:
    def update(self, observable, *args, **kwargs):
        pass

class View(Observer, tk.Tk):

    PAD_WIDGET = 5
    PAD_FRAME = 20
    WIDTH_ENTRY = 40

    def __init__(self, controller):
        super().__init__()
        
        # set the controller for this view
        self.controller = controller

        # set the title and size of the window
        self.title('YOLOv8 Model Application')

        # create the main frame in the window
        self._createMainFrame()
        
    def main(self):
        self.mainloop()

    def _createMainFrame(self):
        self.frm_main = ttk.Frame(self, padding=self.PAD_FRAME)
        self.frm_main.pack(fill="both", padx=self.PAD_FRAME, pady=self.PAD_FRAME)
        self._createLabelFramesAndWidgets()
      
    def _createLabelFramesAndWidgets(self):
        self.modelFilePath1 = tk.StringVar()
        self.modelFilePath2 = tk.StringVar()
        self.modelFilePath3 = tk.StringVar()
        self.modelFilePath4 = tk.StringVar()
        self.modelFilePath5 = tk.StringVar()
        self.stringVars = [self.modelFilePath1, self.modelFilePath2, self.modelFilePath3, self.modelFilePath4, self.modelFilePath5]
        
        # Step 1: Select one or more model files
        self._createSelectModelFilesLabelFrame()
        self._createSelectModelFilesWidgets()
        
        # Step 2: Select the images folder
        self._createSelectImagesFolderLabelFrame()
        self._createSelectImagesFolderWidgets()
        
        # Step 3: Select the output folder
        self._createSelectOutputFolderLabelFrame()
        self._createSelectOutputFolderWidgets()

        # Predict button
        self._createPredictButton()

    def _createSelectModelFilesLabelFrame(self):
        self.frm_step1 = ttk.LabelFrame(self.frm_main, text='Step 1')
        self.frm_step1.pack(fill="both", expand=1, pady=self.PAD_FRAME)
        self.frm_step1.columnconfigure(1, weight=2)

    def _createSelectImagesFolderLabelFrame(self):
        self.frm_step2 = ttk.LabelFrame(self.frm_main, text='Step 2')
        self.frm_step2.pack(fill="both", expand=1, pady=self.PAD_FRAME)
        self.frm_step2.columnconfigure(1, weight=2)

    def _createSelectOutputFolderLabelFrame(self):
        self.frm_step3 = ttk.LabelFrame(self.frm_main, text='Step 3')
        self.frm_step3.pack(fill="both", expand=1, pady=self.PAD_FRAME)
        self.frm_step3.columnconfigure(1, weight=2)

    def _createSelectModelFilesWidgets(self):
        # widgets for selecting one or more YOLOv8 model files
        lbl_selectModelFiles = ttk.Label(self.frm_step1, text='Select your YOLOv8 model(s):')
        lbl_selectModelFiles.grid(row=0, column=0, padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)

        ent_selectModelFile1 = ttk.Entry(self.frm_step1, textvariable=self.modelFilePath1, width=self.WIDTH_ENTRY)
        ent_selectModelFile1.grid(row=0, column=1, sticky='ew', padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)
        btn_selectModelFile1 = ttk.Button(self.frm_step1, text='Browse...', command= lambda : self._selectModelFile(self.modelFilePath1))
        btn_selectModelFile1.grid(row=0, column=2, padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)

        ent_selectModelFile2 = ttk.Entry(self.frm_step1, textvariable=self.modelFilePath2, width=self.WIDTH_ENTRY)
        ent_selectModelFile2.grid(row=1, column=1, sticky='ew', padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)
        btn_selectModelFile2 = ttk.Button(self.frm_step1, text='Browse...', command= lambda : self._selectModelFile(self.modelFilePath2))
        btn_selectModelFile2.grid(row=1, column=2, padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)

        ent_selectModelFile3 = ttk.Entry(self.frm_step1, textvariable=self.modelFilePath3, width=self.WIDTH_ENTRY)
        ent_selectModelFile3.grid(row=2, column=1, sticky='ew', padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)
        btn_selectModelFile3 = ttk.Button(self.frm_step1, text='Browse...', command= lambda : self._selectModelFile(self.modelFilePath3))
        btn_selectModelFile3.grid(row=2, column=2, padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)

        ent_selectModelFile4 = ttk.Entry(self.frm_step1, textvariable=self.modelFilePath4, width=self.WIDTH_ENTRY)
        ent_selectModelFile4.grid(row=3, column=1, sticky='ew', padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)
        btn_selectModelFile4 = ttk.Button(self.frm_step1, text='Browse...', command= lambda : self._selectModelFile(self.modelFilePath4))
        btn_selectModelFile4.grid(row=3, column=2, padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)

        ent_selectModelFile5 = ttk.Entry(self.frm_step1, textvariable=self.modelFilePath5, width=self.WIDTH_ENTRY)
        ent_selectModelFile5.grid(row=4, column=1, sticky='ew', padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)
        btn_selectModelFile5 = ttk.Button(self.frm_step1, text='Browse...', command= lambda : self._selectModelFile(self.modelFilePath5))
        btn_selectModelFile5.grid(row=4, column=2, padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)

    def _createSelectImagesFolderWidgets(self):
        # widgets for selecting a folder with images
        lbl_selectImagesFolder = ttk.Label(self.frm_step2, text='Select the folder with your images:')
        lbl_selectImagesFolder.grid(row=1, column=0, padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)
        self.imagesFolderPath = tk.StringVar()
        ent_selectImagesFolder = ttk.Entry(self.frm_step2, textvariable=self.imagesFolderPath, width=self.WIDTH_ENTRY)
        ent_selectImagesFolder.grid(row=1, column=1, sticky='ew', padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)
        btn_selectImagesFolder = ttk.Button(self.frm_step2, text='Browse...', command=self._selectImagesFolder)
        btn_selectImagesFolder.grid(row=1, column=2, padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)

    def _createSelectOutputFolderWidgets(self):
        # widgets for selecting a folder for the output file
        lbl_selectOutputFolder = ttk.Label(self.frm_step3, text='Select the folder for the output file:')
        lbl_selectOutputFolder.grid(row=2, column=0, padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)
        self.outputFolderPath = tk.StringVar()
        ent_selectOutputFolder = ttk.Entry(self.frm_step3, textvariable=self.outputFolderPath, width=self.WIDTH_ENTRY)
        ent_selectOutputFolder.grid(row=2, column=1, sticky='ew', padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)
        btn_selectOutputFolder = ttk.Button(self.frm_step3, text='Browse...', command=self._selectOutputFolder)
        btn_selectOutputFolder.grid(row=2, column=2, padx=self.PAD_WIDGET, pady=self.PAD_WIDGET)

    def _selectModelFile(self, entry):
        selectedModelPath = filedialog.askopenfilename(filetypes=[('YOLOv8 model', '*.pt')], title='Select your YOLOv8 model')
        if selectedModelPath: # Only update entry when a file is chosen, do nothing when cancelled
            entry.set(selectedModelPath)

    def _selectImagesFolder(self):
        selectedImagesFolderPath = filedialog.askdirectory(title='Select the folder with your images')
        if selectedImagesFolderPath: # Only update entry when a folder is chosen, do nothing when cancelled
            self.imagesFolderPath.set(selectedImagesFolderPath)

    def _selectOutputFolder(self):
        selectedOutputFolderPath = filedialog.askdirectory(title='Select the folder for the output file')
        if selectedOutputFolderPath: # Only update entry when a folder is chosen, do nothing when cancelled
            self.outputFolderPath.set(selectedOutputFolderPath)

    def _createPredictButton(self):
        self.btn_predict = ttk.Button(self.frm_main, text='Predict!', command = self._executePrediction)
        self.btn_predict.pack(fill="both", expand=1, pady=self.PAD_FRAME)

    #---------------------------
    # Other methods
    #---------------------------
    def update(self, observable):
        if isinstance(observable, Model):
            progress_count = observable.progressCount
            print(progress_count)

    def _executePrediction(self):
        modelPathsList = []
        for stringVar in self.stringVars:
            modelPath = stringVar.get()
            if modelPath:
                modelPathsList.append(modelPath)
        imagesPath = self.imagesFolderPath.get()
        outputPath = self.outputFolderPath.get()
        
        self.controller.executePrediction(modelPathsList, imagesPath, outputPath)

    def showErrorMessageBox(self, message):
        messagebox.showerror("Error", message)