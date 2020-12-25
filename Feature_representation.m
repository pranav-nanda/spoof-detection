%% Create an audio data store
trainfolder = fullfile('..','LA_dev'); % the training set of the LA partition is already stored in a folder "LA_train" that contains
                                         % two sub folders namely "bonafide" and "spoof"
ads_train = audioDatastore(trainfolder,...
    'IncludeSubfolders',true,...
    'FileExtensions','.flac',...
    'LabelSource','foldernames');
%% Compute and save the various features of the training set into different folders
i = 1;
labels = ads_train.Labels;
while hasdata(ads_train)
if(i>23569)
[sig,info] = read(ads_train);   % read the speech signal one by one
Fs = info.SampleRate; 
mel = melSpectrogram(sig,Fs,'SpectrumType','power','NumBands',100,'FFTLength',2048);
coeffs =mdwtdec('c',log10(mel+eps),2,'dmey');   % compute the 3 sets of wavelet features
mel_fc = mfcc(sig,Fs,'NumCoeffs',40,'FFTLength',1024); % compute mfcc
%LL = imresize(coeffs.ca,[224 224]);  % resize the LL coefficients 
LH = imresize(coeffs.cd{1,2},[224 224]);  % resize the LH coefficients
H = imresize(coeffs.cd{1,1},[224 224]);  % resize the H coefficients
melfc = imresize(mel_fc,[224 224]);   % resize the mfcc coefficients
%saveFeat(LL,i,labels);      % save the LL features into a folder
saveFeat1(LH,i,labels);     % save the LH features into a folder
saveFeat2(H,i,labels);      % save the H features into a folder
saveFeat3(melfc,i,labels);  % save the mfcc features into a folder
end
i = i + 1;
end 
%% Audio data store for the development set
devfolder = fullfile('..','LA_dev');  % the development set of the LA partition is already stored in a folder "LA_dev" that contains
                                      % two sub folders namely "bonafide" and "spoof"
ads_dev = audioDatastore(devfolder,...
    'IncludeSubfolders',true,...
    'FileExtensions','.flac',...
    'LabelSource','foldernames');
%% Compute and save the various features of the development set
i = 1;
labels_dev = ads_dev.Labels;
while hasdata(ads_dev)
[sig,info] = read(ads_dev);
Fs = info.SampleRate; 
mel = melSpectrogram(sig,Fs,'SpectrumType','power','NumBands',40,'FFTLength',1024);
coeffs =mdwtdec('c',log10(mel+eps),2,'dmey');
mel_fc = mfcc(sig,Fs,'NumCoeffs',40,'FFTLength',1024);
LL = imresize(coeffs.ca,[224 224]);
LH = imresize(coeffs.cd{1,2},[224 224]);
H = imresize(coeffs.cd{1,1},[224 224]);
melfc = imresize(mel_fc,[224 224]);
saveFeat(LL,i,labels_dev);
saveFeat1(LH,i,labels_dev);
saveFeat2(H,i,labels_dev);
saveFeat3(melfc,i,labels_dev);
i = i + 1;
end 
%% Functions to save the Feature representations into different folders
function saveFeat(mel,i,labels1)
folder = 'C:\Users\Saikata\Desktop\LL_train';  % create a folder "LL_train" in the desired location with two sub folders namely "bonafide" and "spoof"
%folder = 'C:\Users\Saikata\Desktop\LL_dev';   % use this folder location for storing development set features
im = ind2rgb(im2uint8(rescale(mel)),jet(512)); % convert to an RGB image by mapping it onto a color map
imgLoc = fullfile(folder,char(labels1(i)));    % specify the sub folder in which the image should be saved (either bonafide or spoof)
imFileName = strcat(char(labels1(i)),'_',num2str(i),'.tif');  % name the image and save it as a TIF file to avoid loss of information
imwrite(im,fullfile(imgLoc,imFileName));
end


function saveFeat1(mel1,k,labels_1)
folder1 = 'C:\Users\Saikata\Desktop\New folder (11)';  % create a folder "LH_train" in the desired location with two sub folders namely "bonafide" and "spoof"
%folder1 = 'C:\Users\Saikata\Desktop\LH_dev';    % use this folder location for storing development set features
im1 = ind2rgb(im2uint8(rescale(mel1)),jet(512));
imgLoc1 = fullfile(folder1,char(labels_1(k)));
imFileName1 = strcat(char(labels_1(k)),'_',num2str(k),'.tif');
imwrite(im1,fullfile(imgLoc1,imFileName1));
end

function saveFeat2(mel2,l,labels_2)
folder2 = 'C:\Users\Saikata\Desktop\New folder (14)';  % create a folder "H_train" in the desired location with two sub folders namely "bonafide" and "spoof"
%folder2 = 'C:\Users\Saikata\Desktop\H_dev';    % use this folder location for storing development set features
im2 = ind2rgb(im2uint8(rescale(mel2,0,0.32)),jet(128));
imgLoc2 = fullfile(folder2,char(labels_2(l)));
imFileName2 = strcat(char(labels_2(l)),'_',num2str(l),'.tif');
imwrite(im2,fullfile(imgLoc2,imFileName2));
end

function saveFeat3(mel3,m,labels_3)
folder3 = 'C:\Users\Saikata\Desktop\New folder (15)';  % create a folder "mfcc_train" in the desired location with two sub folders namely "bonafide" and "spoof"
%folder3 = 'C:\Users\Saikata\Desktop\mfcc_dev';   % use this folder location for storing development set features
im3 = ind2rgb(im2uint8(mel3),jet(128));
imgLoc3 = fullfile(folder3,char(labels_3(m)));
imFileName3 = strcat(char(labels_3(m)),'_',num2str(m),'.tif');
imwrite(im3,fullfile(imgLoc3,imFileName3));
end