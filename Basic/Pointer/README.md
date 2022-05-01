# The Power of Pointer Networks 

In this project we showcase the ability of pointer networks to extract information from semi
-structured documents on a toy dataset.

This folder contains five files/directory:

- documents.json: ground truth for the toy dataset
- images: folder containing the images of the documents, NB all the data is fake and generated
 with `Faker`
- model.py: `pytorch` implementation of our model  
- utils.py: functions used within the notebook
- information_extraction_with_pointer_network.ipynb: notebook with the whole flow:
    1. Optical Character Recognition
    2. ground truth pre-processing
    3. Data pre-processing
    4. Model training
    5. Model results 