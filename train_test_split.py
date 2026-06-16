import os
import splitfolders

# Define the path to the main folder containing the images
main_folder = r"C:\Users\fhesn\OneDrive\Masaüstü\domates_hastalik"


#Start the train test split

splitfolders.ratio(main_folder, 
                   output=r"C:\Users\fhesn\OneDrive\Masaüstü\PlantVillage_split",
                     seed=1337, ratio=(.8, .2))
