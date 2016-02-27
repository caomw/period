clear all; close all;
addpath('../../marvin/tools/tensorIO_matlab');

dataDir = '../../../data/train';
dstDir = '../../../data/tensors';
mkdir(dstDir);

data2DFilename = 'data2DTrain.tensor';
data3DFilename = 'data3DTrain.tensor';
labelsFilename = 'labelsTrain.tensor';

% Change random seed
rng(sum(100*clock),'twister');

% List files
colorFiles = dir(fullfile(dataDir,'*.color.png'));
cropFiles = dir(fullfile(dataDir,'*.crop.txt'));
tsdfFiles = dir(fullfile(dataDir,'*.tsdf.bin'));

% Count number of training data points
num_data = 0;
for fileIDX=1:length(cropFiles)
    cropInfo = dlmread(fullfile(dataDir,cropFiles(fileIDX).name));
    num_data = num_data + sum(cropInfo(:,1) > 0);
end

data2D = zeros(227,227,3,num_data);
data3D = zeros(30,30,30,1,num_data);
labels = zeros(1,1,1,1,num_data);
dataIDX = 1;
for fileIDX=1:length(colorFiles)
    fprintf('%d/%d\n',fileIDX,length(colorFiles));
    filename = colorFiles(fileIDX).name(1:(end-10));

    % Load color image
    I = imread(fullfile(dataDir,colorFiles(fileIDX).name));
    
    % Load TSDF volume
    delimIDX = strfind(tsdfFiles(fileIDX).name,'.');
    tsdfDimX = str2double(tsdfFiles(fileIDX).name((delimIDX(1)+1):(delimIDX(2)-1)));
    tsdfDimY = str2double(tsdfFiles(fileIDX).name((delimIDX(2)+1):(delimIDX(3)-1)));
    tsdfDimZ = str2double(tsdfFiles(fileIDX).name((delimIDX(3)+1):(delimIDX(4)-1)));
    fileID = fopen(fullfile(dataDir,tsdfFiles(fileIDX).name),'r');
    tsdf = fread(fileID,'single');
    fclose(fileID);
    tsdf = reshape(tsdf,tsdfDimX,tsdfDimY,tsdfDimZ);
    
    % Load crop info
    cropInfo = dlmread(fullfile(dataDir,cropFiles(fileIDX).name));
%     cropInfo(:,2:5) = round(cropInfo(:,2:5));
    cropInfo(:,2:end) = cropInfo(:,2:end) + 1;
    
    posIDX = find(cropInfo(:,1) > 0);
    negIDX = find(cropInfo(:,1) == 0);
    negIDX = randsample(negIDX,size(posIDX,1));
    
    for i = 1:size(posIDX,1)
        % Create 2D image patch and 3D tsdf cube for positive case
        cubeCrop = cropInfo(posIDX(i),:);
        patch = I(cubeCrop(4):cubeCrop(5),cubeCrop(2):cubeCrop(3),:);
        patch = imresize(patch,[227,227]);
        patch = double(patch(:,:,[3,2,1]));
        patch(:,:,1) = patch(:,:,1) - 102.9801;
        patch(:,:,2) = patch(:,:,2) - 115.9465;
        patch(:,:,3) = patch(:,:,3) - 122.7717;
        data2D(:,:,:,dataIDX) = patch;
        cube = tsdf(cubeCrop(6):cubeCrop(7),cubeCrop(8):cubeCrop(9),cubeCrop(10):cubeCrop(11));
        data3D(:,:,:,:,dataIDX) = cube;
        labels(:,:,:,:,dataIDX) = 1;
        dataIDX = dataIDX + 1;
        
        % Create 2D image patch and 3D tsdf cube for negative case
        cubeCrop = cropInfo(negIDX(i),:);
        patch = I(cubeCrop(4):cubeCrop(5),cubeCrop(2):cubeCrop(3),:);
        patch = imresize(patch,[227,227]);
        patch = double(patch(:,:,[3,2,1]));
        patch(:,:,1) = patch(:,:,1) - 102.9801;
        patch(:,:,2) = patch(:,:,2) - 115.9465;
        patch(:,:,3) = patch(:,:,3) - 122.7717;
        data2D(:,:,:,dataIDX) = patch;
        cube = tsdf(cubeCrop(6):cubeCrop(7),cubeCrop(8):cubeCrop(9),cubeCrop(10):cubeCrop(11));
        data3D(:,:,:,:,dataIDX) = cube;
        labels(:,:,:,:,dataIDX) = 0;
        dataIDX = dataIDX + 1;
    end
    
%     imshow(I(cubeCrop(4):cubeCrop(5),cubeCrop(2):cubeCrop(3),:))
%     points = [];
%     for x = 1:30
%         for y=1:30
%             for z=1:30
%                 if abs(cube(x,y,z)) < 0.2
%                     points = [points; x, y, z];
%                 end
%                 edge = 0;
%                 if x == 1 || x == 30
%                     edge = edge + 1;
%                 end
%                 if y == 1 || y == 30
%                     edge = edge + 1;
%                 end
%                 if z == 1 || z == 30
%                     edge = edge + 1;
%                 end
%                 if edge > 1
%                     points = [points; x, y, z];
%                 end
%             end
%         end
%     end
%     ptCloud = pointCloud(points);
%     pcwrite(ptCloud,'test.ply','PLYFormat','binary');
    
    
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