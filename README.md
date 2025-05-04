# Instructions for Running the Code for Eatalyze: Using Machine Learning to Classify and Analyze Food Nutrition


## Linear Regression and XGBOOST Scripts
#### pip install the following packages:
1.     pip install numpy
2.     pip install pandas
3.     pip install matplotlib
4.     pip install scikit-learn
5.     pip install statsmodels
6.     pip install xgboost

Ensure the [food.xlsx](https://www.kaggle.com/datasets/shrutisaxena/food-nutrition-dataset) data is installed and in the same folder as your jupyter notebooks for this line to run
`food = pd.read_excel("food.xlsx", sheet_name="food")`. From there, just run all cells and the graphs and tables should appear!



## ML_Custom_Classifier.py
Below are instructions to run the **training script**, which will train the model and save the parameters to a file on your computer. This section will guide you on how to download the food-101 images and install the required packages. 

#### pip install the following packages:
1.     pip install numpy
2.     pip install pandas
3.     pip install matplotlib
4.     pip install scikit-learn
5.     pip install tensorflow

#### To download the food-101 dataset:

    import kagglehub
    path = kagglehub.dataset_download("kmader/food41")
    print("Path to dataset files:", path)

#### Now run the program:
- Update the dataset path in the program to the path on your specific computer
- Run this updated script and ensure that the Epochs are incrementing
- Watch grass grow : )


The **ML_Image_Classifier.py** script opens a new window and uses the computer camera to classify objects in front of it. This uses the model that was previously trained with __ML_Custom_Classifier.py__. This will also require a few packages to run.

#### pip install the following packages:
1.     pip install numpy
2.     pip install pandas
3.     pip install matplotlib
4.     pip install scikit-learn
5.     pip install tensorflow
6.     pip install opencv-python

#### Ensure the model is correct:
    model = tf.keras.models.load_model("food_cnn_model_custom.h5")

#### Now run the program:
- A new window will open using the built-in laptop camera
- When the desired food is within this frame, press the "s" key to capture an image
- This image will then be passed through the classifier to predict the image's **label**
- The resulting prediction will output to the terminal with an associated confidence level
- Press the "q" key to quit the program and exit
