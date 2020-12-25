# Before executing this script two sub folders within the "ASVspoof2019_LA_train" folder needs to be created. Namely "bonafide" and "spoof"
# The same script can be used to seggregate audio files from the development and evaluation sets

import os        #used to travel through the file system and control os
import shutil as sh   #used to do stuff like move, copy, rename


#open the text file
with open ("C:\\Users\\Saikata\\Downloads\\LA\\LA\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.train.trn.txt", 'r') as file:
	#go through each line
        for line in file:
		#split makes a list of strings by removing where ever there is space in the line
                name=line.split()[1]
                category=line.split()[-1]
                #use move method in sh to move from one folder to the other
                sh.move(f"C:\\Users\\Saikata\\Downloads\\LA\\LA\\ASVspoof2019_LA_train\\flac\\{name}.flac",
                        f"C:\\Users\\Saikata\\Downloads\\LA\\LA\\ASVspoof2019_LA_train\\{category}\\{name}.flac")
