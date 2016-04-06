% Code for SIFT-based 3D reconstruction with RGB-D frames

%% Load directories and file locations
% Directory containing sequence of RGB-D frames
% RGB-D frame format:
%     Color: frame-XXXXXX.color.png (640x480, RGB-8, 24-bit, PNG)
%     Depth: frame-XXXXXX.depth.png (640x480, depth in millimeters, 16-bit, PNG)
% Computed extrinsics will be saved to the same directory
seq = 'data/glue.test';

% Load intrinsic matrix for depth camera
K = dlmread(fullfile(seq,'intrinsics.K.txt'));

% List frames
depthFiles = dir(fullfile(seq,'*.depth.png'));
colorFiles = dir(fullfile(seq,'*.color.png'));

% Load libraries
addpath(fullfile('lib','vlfeat','toolbox','mex','mexa64'));
addpath(fullfile('lib','vlfeat','toolbox','mex','mexw64'));
addpath(fullfile('lib','peter'));
addpath(fullfile('lib','estimateRigidTransform'));
addpath(fullfile('lib','icp'));

%% Loop through each frame and collect SIFT keypoints and descriptors
numFrames = length(depthFiles);
allSIFTpts = cell(1,numFrames);
allSIFTdes = cell(1,numFrames);
allFramePts = cell(1,numFrames);
parfor frameIDX = 1:numFrames
    fprintf('Computing SIFT keypoints from frame %d/%d\n',frameIDX,length(depthFiles));
    
    % Load RGB-D frame
    image = imread(fullfile(seq,colorFiles(frameIDX).name));
    depth = double(imread(fullfile(seq,depthFiles(frameIDX).name)))./1000;
    
    % Set invalid depth to 0
    depth(find(depth < 0.2)) = 0;
    depth(find(depth > 0.8)) = 0;
    
    % Fill holes in depth using local max
    depth = fillHolesDepth(depth);
    
    % Compute SIFT keypoints and descriptors
    [currSIFTpts,currSIFTdes] = up_sift(single(rgb2gray(image)));  
    
    % Use only SIFT keypoints with valid depth values
    SIFTind = sub2ind(size(depth),round(currSIFTpts(2,:))',round(currSIFTpts(1,:))');
    validSIFTind = find(depth(SIFTind) > 0);
    currSIFTpts = currSIFTpts(:,validSIFTind);
    currSIFTdes = currSIFTdes(:,validSIFTind);
    
    % Convert SIFT keypoints to 3D
    SIFTptsZ = depth(SIFTind(validSIFTind));
    SIFTptsX = (round(currSIFTpts(1,:))'-K(1,3)).*SIFTptsZ/K(1,1);
    SIFTptsY = (round(currSIFTpts(2,:))'-K(2,3)).*SIFTptsZ/K(2,2);
    SIFTcamXYZ = [SIFTptsX, SIFTptsY, SIFTptsZ]';
    
    % Compute XYZ camera points
    [pixX,pixY] = meshgrid(1:640,1:480);
    camX = (pixX-K(1,3)).*depth/K(1,1);
    camY = (pixY-K(2,3)).*depth/K(2,2);
    camZ = depth;
    validPtsIDX = find(camZ > 0);
    currFramePts = cat(2,camX(validPtsIDX),camY(validPtsIDX),camZ(validPtsIDX))';
    
    % Save SIFT keypoints, descriptors, camera points to structure
    allSIFTpts{frameIDX} = SIFTcamXYZ;
    allSIFTdes{frameIDX} = currSIFTdes;
    allFramePts{frameIDX} = currFramePts;
end

% Loop through each frame and compute its extrinsics using SIFT
for frameIDX = 1:numFrames
    fprintf('Computing extrinsics for frame %d/%d\n',frameIDX,length(depthFiles));

    % First frame is set to identity
    if frameIDX == 1
       RT = [eye(3),[0;0;0]];
       validthisframe = 1;
       
    % Estimate extrinsics using SIFT + RANSAC + ICP
    else
        error3D_threshold = 0.005;
        matchSIFT_threshold = 8;
        matchSIFT_ratioTest = 0.7^2;
%         [RT,NumMatch] = robustAlignRt(allPrevSIFTDes,allSIFTdes{frameIDX},...
%                                       allPrevSIFTPts,allSIFTpts{frameIDX},...
%                                       allPrevFramePts,allFramePts{frameIDX},...
%                                       matchSIFT_threshold,matchSIFT_ratioTest,error3D_threshold);
        
        % Concatenate all SIFT information from previous k frames
        prevK = numFrames;
        allPrevSIFTPts = [];
        allPrevSIFTDes = [];
        allPrevFramePts = [];
        for prevFrameIDX = max(1,frameIDX-prevK):frameIDX-1
            allPrevSIFTPts = [allPrevSIFTPts allSIFTpts{prevFrameIDX}];
            allPrevSIFTDes = [allPrevSIFTDes allSIFTdes{prevFrameIDX}];
            allPrevFramePts = [allPrevFramePts allFramePts{prevFrameIDX}];
        end
        [RT,NumMatch] = robustAlignRt(allPrevSIFTDes,allSIFTdes{frameIDX},...
                                      allPrevSIFTPts,allSIFTpts{frameIDX},...
                                      allPrevFramePts,allFramePts{frameIDX},...
                                      matchSIFT_threshold,matchSIFT_ratioTest,error3D_threshold);
    end
    
    % Save extrinsics
    RT = [RT; 0 0 0 1];
    frameName = fullfile(seq,colorFiles(frameIDX).name);
    frameName = frameName(1:(end-10));
    fileID = fopen(strcat(frameName,'.pose.txt'),'w');
    for i = 1:4
        fprintf(fileID, '%.17g %.17g %.17g %.17g\n',RT(i,1),RT(i,2),RT(i,3),RT(i,4));
    end
    fclose(fileID);
end

%% BACKUP Code
% % Loop through each frame and compute its extrinsics
% SIFTdata = [];
% densePtCloud =[];
% extrinsicsC2W =[];
% mkdir('data');
% for frameIDX = 1:length(depthFiles)
%     fprintf('Aligning frame %d/%d\n',frameIDX,length(depthFiles));
%     
%     % Load RGB-D frame
%     I = imread(fullfile(seq,colorFiles(frameIDX).name));
%     D = double(imread(fullfile(seq,depthFiles(frameIDX).name)))./1000;
%     
% %     % Make sure depth is not corrupted
% %     if sum(find(D > 0)) == 0
% %         delete(fullfile(seq,colorFiles(frameIDX).name));
% %         delete(fullfile(seq,depthFiles(frameIDX).name));
% %         continue;
% %     end
%     
%     % Set invalid depth to 0
%     D(find(D < 0.2)) = 0;
%     D(find(D > 0.8)) = 0;
%     
%     % Make depth frame denser
%     D = propagateDepth(D);
%     
%     % Get XYZ camera coordinates
%     [pixX,pixY] = meshgrid(1:640,1:480);
%     camX = (pixX-K(1,3)).*D/K(1,1);
%     camY = (pixY-K(2,3)).*D/K(2,2);
%     camZ = D;
%     camXYZ = cat(3,camX,camY,camZ,(camZ > 0));
%     
%     % Compute SIFT keypoints and descriptors
%     [SIFTpts,SIFTdes] = up_sift(single(rgb2gray(I)));  
%     selectedMask = camXYZ(:,:,4) > 0;
%     inMask = SiftInMask(SIFTpts,selectedMask);
% 
%     % Use only valid SIFT keypoints
%     SIFTpts = SIFTpts(:,inMask);
%     SIFTdes = SIFTdes(:,inMask);
%     
%     % Project SIFT keypoints to camera coordinates
%     [valid, P3D] = get3DforSIFT(camXYZ,SIFTpts);  
%     SIFTpts = SIFTpts(:,valid);
%     SIFTdes = SIFTdes(:,valid);
%     
%     % Project all points to camera coordinates
%     ptCloud = get3Dpoints(camXYZ,selectedMask);
%     
%     % First frame is set to identity
%     if frameIDX == 1
%        RT = [eye(3),[0;0;0]];
%        validthisframe = 1;
%        
%     % Estimate extrinsics with RANSAC 
%     else
%        error3D_threshold = 0.005;
%        matchSIFT_threshold = 8;
%        matchSIFT_ratioTest = 0.7^2;
%        [RT,NumMatch] = robustAlignRt([SIFTdata.des],SIFTdes,...
%                                      [SIFTdata.P3D],P3D,...
%                                      cell2mat(densePtCloud),ptCloud,...
%                                      matchSIFT_threshold,matchSIFT_ratioTest,error3D_threshold);
%         validthisframe = NumMatch.numinlier/NumMatch.nummatch>0.7;
%     end
%     
%     % Save extrinsics
%     RT = [RT; 0 0 0 1];
% %     if (frameIDX > 1)
% %         extrinsicsC2W{frameIDX} = RT*extrinsicsC2W{frameIDX-1};
% %     else
% %         extrinsicsC2W{frameIDX} = RT;
% %     end
%     extrinsicsC2W{frameIDX} = RT;
% %     RT = extrinsicsC2W{frameIDX};
%     frameName = fullfile(seq,colorFiles(frameIDX).name);
%     frameName = frameName(1:(end-10));
%     fileID = fopen(strcat(frameName,'.pose.txt'),'w');
%     for i = 1:4
%         fprintf(fileID, '%.17g %.17g %.17g %.17g\n',RT(i,1),RT(i,2),RT(i,3),RT(i,4));
%     end
%     fclose(fileID);
%     
%     % Save SIFT history data
%     SIFTdata(frameIDX).pts = SIFTpts;
%     SIFTdata(frameIDX).des = SIFTdes;
%     SIFTdata(frameIDX).P3D = transformPointCloud(P3D,extrinsicsC2W{frameIDX});
%     densePtCloud{frameIDX} = transformPointCloud(ptCloud,extrinsicsC2W{frameIDX});
%     frameIds(frameIDX) = frameIDX;
%     points = densePtCloud{frameIDX};
%     colors = get3Dpoints(I, selectedMask);
%     
% %     % Save point cloud
% %     pcwrite(pointCloud(points','Color',colors'),fullfile('data',sprintf('ptCloud%d',frameIDX)),'PLYFormat','binary');
% 
% end

%% BACKUP Code
% I = imread('../../data/seq/frame-000000.color.png');
% D = double(imread('../../data/seq/frame-000000.depth.png'))./1000;
% 
% [x,y] = meshgrid(1:640,1:480);
% camX = (x-K(1,3)).*D/K(1,1);
% camY = (y-K(2,3)).*D/K(2,2);
% camZ = D;
% camXYZ = [camX(:) camY(:) camZ(:)]';
% 
% r = I(:,:,1);
% g = I(:,:,2);
% b = I(:,:,3);
% colors = [r(:) g(:) b(:)]';
% 
% ptCloud = pointCloud(camXYZ','Color',colors');
% pcwrite(ptCloud,'test','PLYFormat','binary');





