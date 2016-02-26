
objCoords = [];
objColors = [];

seqDir = '../../data/train1';

% List RGB-D frames
colorFiles = dir(fullfile(seqDir,'*.color.png'));
depthFiles = dir(fullfile(seqDir,'*.depth.png'));

% Load intrinsics
K = dlmread(fullfile(seqDir,'intrinsics.K.txt'));

for frameIDX=1:length(depthFiles)
    fprintf('Loading frame %d/%d\n',frameIDX,length(depthFiles));
    
    % Load RGB-D frame
    I = imread(fullfile(seqDir,colorFiles(frameIDX).name));
    D = double(imread(fullfile(seqDir,depthFiles(frameIDX).name)))./1000;
    
    % Only load valid depth
    D(find(D < 0.2)) = 0;
    D(find(D > 0.8)) = 0;
    
    % Get XYZ camera coordinates
    [pixX,pixY] = meshgrid(1:640,1:480);
    camX = (pixX-K(1,3)).*D/K(1,1); 
    camY = (pixY-K(2,3)).*D/K(2,2); 
    camZ = D;
    camXYZ = [camX(:) camY(:) camZ(:)]';
    
%     % Load only valid XYZ points
    validIDX = find(camXYZ(3,:) > 0);
    camXYZ = camXYZ(:,validIDX);
    
    % Apply extrinsics
    frameName = colorFiles(frameIDX).name(1:(end-10));
    ext = dlmread(fullfile(seqDir,sprintf('%s.pose.txt',frameName)));
    camXYZ = ext(1:3,1:3)*camXYZ + repmat(ext(1:3,4),1,size(camXYZ,2));
    
    % Get point colors
    colorsR = I(:,:,1);
    colorsG = I(:,:,2);
    colorsB = I(:,:,3);
    colors = cat(1,colorsR(:)',colorsG(:)',colorsB(:)');
    colors = colors(:,validIDX);
    
    % Save XYZ coords and colors
    objCoords = [objCoords,camXYZ];
    objColors = [objColors,colors];
end

% Save point cloud to file
ptCloud = pointCloud(objCoords','Color',objColors');
pcwrite(ptCloud,'object','PLYFormat','binary');

% Prompt the user to manually create bounding box
fprintf('Open the PLY file in Meshlab and remove all non-object points.\nPress any key to continue...\n');
pause;

% Load point cloud
ptCloud = pcread(sprintf('%s.ply','object'));

% Get object's 3D bbox from point cloud
bbox = [ptCloud.XLimits; ptCloud.YLimits; ptCloud.ZLimits];
save('bbox.mat','bbox');


% % Create masks for each frame using object's 3D bbox
% for frameIDX=1:100
%     frameIDX
%     load(fullfile(objDir,sprintf('%d.mat',frameIDX)));
%     
%     % Calibrate data
%     [tmpImageUncalib,tmpXYZcameraUncalib] = readData_realsense(rgbd_points_data);
%     [tmpImage, tmpXYZcamera] = calibrateData_realsense(cameraRGBImage_data, tmpXYZcameraUncalib);
%     
%     % Get all points within bounding box
%     objMaskIndLayers = logical(zeros(720*1280,6));
%     objMaskIndLayers(find(tmpXYZcamera(:,:,1) > bbox(1,1)),1) = 1;
%     objMaskIndLayers(find(tmpXYZcamera(:,:,1) < bbox(1,2)),2) = 1;
%     objMaskIndLayers(find(tmpXYZcamera(:,:,2) > bbox(2,1)),3) = 1;
%     objMaskIndLayers(find(tmpXYZcamera(:,:,2) < bbox(2,2)),4) = 1;
%     objMaskIndLayers(find(tmpXYZcamera(:,:,3) > bbox(3,1)),5) = 1;
%     objMaskIndLayers(find(tmpXYZcamera(:,:,3) < bbox(3,2)),6) = 1;
%     objMaskInd = objMaskIndLayers(:,1) & objMaskIndLayers(:,2) & objMaskIndLayers(:,3) & ...
%                  objMaskIndLayers(:,4) & objMaskIndLayers(:,5) & objMaskIndLayers(:,6);
%     
%     % Get object mask and save to file
%     objMask = reshape(objMaskInd,720,1280);
%     imwrite(objMask,fullfile(objDir,sprintf('%d.jpg',frameIDX)));
%     
% end



