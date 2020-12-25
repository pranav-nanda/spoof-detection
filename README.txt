The code developed here requires the following toolboxes in MATLAB
--Deep learning toolbox
--Wavelet toolbox
--Audio toolbox
--Signal processing toolbox
--Image processing toolbox
--Deep Learning Toolbox Model for ResNet-50 Network
(link: https://www.mathworks.com/matlabcentral/fileexchange/64626-deep-learning-toolbox-model-for-resnet-50-network)

(The use of GPU may require some additional toolboxes)

1) When the Logical Access (LA) dataset is downloaded, the audio files contained in the folders "ASVspoof2019_LA_train",
  "ASVspoof2019_LA_dev" and "ASVspoof2019_LA_eval" are jumbled up. The "ASVspoof2019_LA_cm_protocols" folder contains 
   the labels of the audio files present in these three folders. The python script titled "Seggregation_script"
   seggregates these jumbled up audio files into two seperate folders namely "bonafide" and "spoof". A python IDLE 
   environment is sufficient to run this script. This is done to facilitate the creation of AudioDataStores in MATLAB.

2) The MATLAB script "Feature_representation" computes the Wavelet and MFCC features and stores them in the required 
   folders as RGB images.

3) The MATLAB script "ResNet" is used for transfer learning by employing the ResNet50 architecture. Final output of this
   script is the normalized scores of the CNN on either the development or evaluation sets.

4) Finally, the script "tDCF_EER" computes the tandem detection cost function and Equal error rates by making use of the
   scores computed in the previous step. Largely the code provided by the organizers is used for this purpose except for
   the error plots and some pre-processing.


