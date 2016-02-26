clear all; close all;
addpath('../../marvin/tools/tensorIO_matlab');

dataDir = 'data/processed/train';
dstDir = 'data/tensors';
data2DFilename = 'data2DTrain.tensor';
data3DFilename = 'data3DTrain.tensor';
labelsFilename = 'labelsTrain.tensor';

% List files
colorFiles = dir(fullfile(dataDir,'*.color.png'));

data2D = zeros(224,224,3,length(colorFiles));
data3D = zeros(30,30,30,3,length(colorFiles));
labels = zeros(1,1,1,1,length(colorFiles));
for i=1:length(colorFiles)
    fprintf('%d/%d\n',i,length(colorFiles));
    filename = colorFiles(i).name(1:(end-10));
    
    % Load rgb data
    I = imread(fullfile(dataDir,strcat(filename,'.color.png')));
    I = imresize(I,[224,224]);
    I = double(I(:,:,[3,2,1]));
    I(:,:,1) = I(:,:,1) - 102.9801;
    I(:,:,2) = I(:,:,2) - 115.9465;
    I(:,:,3) = I(:,:,3) - 122.7717;
    data2D(:,:,:,i) = I;
    
    % Load tsdf data
    load(fullfile(dataDir,strcat(filename,'.tsdf.mat')));
    data3D(:,:,:,:,i) = tsdf;
    
    if strcmp(filename(1:3),'pos')
        labels(:,:,:,:,i) = 1;
    else
        labels(:,:,:,:,i) = 0;
    end
    
end

% Create data2D tensor
data2DTensor.dim = 4;
data2DTensor.name = 'data2D';
data2DTensor.type = 'float';
data2DTensor.value = data2D;
data2DTensor.sizeof = 4;
writeTensors(fullfile(dstDir,data2DFilename),data2DTensor);

% Create data3D tensor
data3DTensor.dim = 5;
data3DTensor.name = 'data3D';
data3DTensor.type = 'float';
data3DTensor.value = data3D;
data3DTensor.sizeof = 4;
writeTensors(fullfile(dstDir,data3DFilename),data3DTensor);

% Create labels tensor
labelsTensor.dim = 5;
labelsTensor.name = 'labels';
labelsTensor.type = 'float';
labelsTensor.value = labels;
labelsTensor.sizeof = 4;
writeTensors(fullfile(dstDir,labelsFilename),labelsTensor);