function demo(n)

%% Demo for SIFT-based 3D reconstruction with RGB-D frames

% Directory containing sequence of RGB-D frames
% RGB-D frame format:
%     Color: frame-XXXXXX.color.png (640x480, RGB-8, 24-bit, PNG)
%     Depth: frame-XXXXXX.depth.png (640x480, depth in millimeters, 16-bit, PNG)
% Computed extrinsics will be saved to the same directory
seq = ['data/glue.train.',num2str(n)];

% Load intrinsic matrix for depth camera
K = dlmread(fullfile(seq,'intrinsics.K.txt'));

% List frames
depthFiles = dir(fullfile(seq,'*.depth.png'));
colorFiles = dir(fullfile(seq,'*.color.png'));

% Load libraries
addpath(fullfile('lib','vlfeat','toolbox','mex','mexa64'));
addpath(fullfile('lib','peter'));
addpath(fullfile('lib','estimateRigidTransform'));
addpath(fullfile('lib','icp'));

% Loop through each frame and compute its extrinsics
SIFTdata = [];
densePtCloud =[];
extrinsicsC2W =[];
mkdir('data');
for frameIDX = 1:length(depthFiles)
    fprintf('Aligning frame %d/%d\n',frameIDX,length(depthFiles));
    
    % Load RGB-D frame
    I = imread(fullfile(seq,colorFiles(frameIDX).name));
    D = double(imread(fullfile(seq,depthFiles(frameIDX).name)))./1000;
    
%     % Make sure depth is not corrupted
%     if sum(find(D > 0)) == 0
%         delete(fullfile(seq,colorFiles(frameIDX).name));
%         delete(fullfile(seq,depthFiles(frameIDX).name));
%         continue;
%     end
    
    % Set invalid depth to 0
    D(find(D < 0.2)) = 0;
    D(find(D > 0.8)) = 0;
    
    % Make depth frame denser
    D = propagateDepth(D);
    
    % Get XYZ camera coordinates
    [pixX,pixY] = meshgrid(1:640,1:480);
    camX = (pixX-K(1,3)).*D/K(1,1);
    camY = (pixY-K(2,3)).*D/K(2,2);
    camZ = D;
    camXYZ = cat(3,camX,camY,camZ,(camZ > 0));
    
    % Compute SIFT keypoints and descriptors
    [SIFTpts,SIFTdes] = up_sift(single(rgb2gray(I)));  
    selectedMask = camXYZ(:,:,4) > 0;
    inMask = SiftInMask(SIFTpts,selectedMask);

    % Use only valid SIFT keypoints
    SIFTpts = SIFTpts(:,inMask);
    SIFTdes = SIFTdes(:,inMask);
    
    % Project SIFT keypoints to camera coordinates
    [valid, P3D] = get3DforSIFT(camXYZ,SIFTpts);  
    SIFTpts = SIFTpts(:,valid);
    SIFTdes = SIFTdes(:,valid);
    
    % Project all points to camera coordinates
    ptCloud = get3Dpoints(camXYZ,selectedMask);
    
    % First frame is set to identity
    if frameIDX == 1
       RT = [eye(3),[0;0;0]];
       validthisframe = 1;
       
    % Estimate extrinsics with RANSAC 
    else
       error3D_threshold = 0.005;
       matchSIFT_threshold = 8;
       matchSIFT_ratioTest = 0.7^2;
       [RT,NumMatch] = robustAlignRt([SIFTdata.des],SIFTdes,...
                                     [SIFTdata.P3D],P3D,...
                                     cell2mat(densePtCloud),ptCloud,...
                                     matchSIFT_threshold,matchSIFT_ratioTest,error3D_threshold);
        validthisframe = NumMatch.numinlier/NumMatch.nummatch>0.7;
    end
    
    % Save extrinsics
    RT = [RT; 0 0 0 1];
%     if (frameIDX > 1)
%         extrinsicsC2W{frameIDX} = RT*extrinsicsC2W{frameIDX-1};
%     else
%         extrinsicsC2W{frameIDX} = RT;
%     end
    extrinsicsC2W{frameIDX} = RT;
%     RT = extrinsicsC2W{frameIDX};
    frameName = fullfile(seq,colorFiles(frameIDX).name);
    frameName = frameName(1:(end-10));
    fileID = fopen(strcat(frameName,'.pose.txt'),'w');
    for i = 1:4
        fprintf(fileID, '%.17g %.17g %.17g %.17g\n',RT(i,1),RT(i,2),RT(i,3),RT(i,4));
    end
    fclose(fileID);
    
    % Save SIFT history data
    SIFTdata(frameIDX).pts = SIFTpts;
    SIFTdata(frameIDX).des = SIFTdes;
    SIFTdata(frameIDX).P3D = transformPointCloud(P3D,extrinsicsC2W{frameIDX});
    densePtCloud{frameIDX} = transformPointCloud(ptCloud,extrinsicsC2W{frameIDX});
    frameIds(frameIDX) = frameIDX;
    points = densePtCloud{frameIDX};
    colors = get3Dpoints(I, selectedMask);
    
%     % Save point cloud
%     pcwrite(pointCloud(points','Color',colors'),fullfile('data',sprintf('ptCloud%d',frameIDX)),'PLYFormat','binary');
end


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





% 
% mkdir(fullfile(datapath,objnames{objectID},'pts'));
% sift = [];
% pcDense =[];
% extrinsicsC2W =[];
% frameIds =[];
% frameIDX =0;
% for frameid =1:numofTraningFrame
% 
%     % read image from training : need to change to multiview   
%     filepath = fullfile(datapath,objnames{objectID},[num2str(frameid) '.mat']);
%     %[imTrain,XYZcam] = readData_realsense(filepath);
%     [~,XYZcameraTestUncalib,cameraRGBImage_data] = readData_realsense(filepath);
% 
%     % Calibrate data from sensor
%     [imTrain, XYZcam] = calibrateData_realsense(cameraRGBImage_data, XYZcameraTestUncalib);
% 
% 
%     %objLabel = imread(fullfile(datapath,objnames{objectID},[num2str(frameid) '.jpg']));
%     selectedMask = XYZcam(:,:,4)>0&XYZcam(:,:,3)<0.4;%&double(objLabel(:,:,1))>100;
%     [loc,des] = up_sift(single(rgb2gray(imTrain)));     
% 
%     inMask = SiftInMask(loc,selectedMask);
% 
%     loc = loc(:,inMask);
%     des = des(:,inMask);
%     [valid, P3D] = get3DforSIFT(XYZcam,loc);  
%     loc = loc(:,valid);
%     des = des(:,valid);
%     pc = get3Dpoints(XYZcam,selectedMask);
%     % align different frames
%     if frameid ==1
%        Rt = [eye(3),[0;0;0]];
%        validthisframe = 1;
%     else
%        error3D_threshold = 0.005;
%        matchSIFT_threshold = 8;
%        matchSIFT_ratioTest = 0.7^2;
%        [Rt,NumMatch] = robustAlignRt([sift.des],des,...
%                                               [sift.P3D],P3D,...
%                                               cell2mat(pcDense),pc,...
%                                               matchSIFT_threshold,matchSIFT_ratioTest,error3D_threshold);
%         validthisframe = NumMatch.numinlier/NumMatch.nummatch>0.7;
%     end
% 
%     frameIDX = frameIDX+1;
%     extrinsicsC2W{frameIDX} = Rt;
%     sift(frameIDX).loc = loc;
%     sift(frameIDX).des = des;
%     sift(frameIDX).P3D = transformPointCloud(P3D,extrinsicsC2W{frameIDX});
%     pcDense{frameIDX} = transformPointCloud(pc,extrinsicsC2W{frameIDX});
%     frameIds(frameIDX) = frameid;
%     pts = pcDense{frameIDX};
%     rgb = get3Dpoints(imTrain, selectedMask);
%     % output ModelPLy
%     %points2ply( fullfile(datapath,objnames{objectID},'pts',['s_' num2str(frameid) '.ply']), pts, repmat(rand(1,3)*255,[length(pts),1]));
%     points2ply( fullfile(datapath,objnames{objectID},'pts',['c_' num2str(frameid) '.ply']), pts, rgb)
% 
% end
% 
% 
% objectModel{objectID}.sift.loc = [sift.loc];
% objectModel{objectID}.sift.des = [sift.des];
% objectModel{objectID}.sift.P3D = [sift.P3D];
% objectModel{objectID}.XYZ =  cell2mat(pcDense);
% objectModel{objectID}.extrinsicsC2W = extrinsicsC2W;
% objectModel{objectID}.frameIds =frameIds;
% 
% % find 
% filepath = fullfile(datapath,objnames{objectID},[num2str(1) '.mat']);
% [imTrain,XYZcam] = readData_realsense(filepath);
% [Rtilt,R] = rectify(objectModel{objectID}.XYZ);
% 
% objectModel{objectID}.XYZ      = [R*objectModel{objectID}.XYZ];
% objectModel{objectID}.sift.P3D = [R*objectModel{objectID}.sift.P3D];
% objectModel{objectID}.R        = R;
% T = 2;
% inrange = objectModel{objectID}.XYZ(1,:)>prctile(objectModel{objectID}.XYZ(1,:),T)&...
%           objectModel{objectID}.XYZ(1,:)<prctile(objectModel{objectID}.XYZ(1,:),100-T)&...
%           objectModel{objectID}.XYZ(2,:)>prctile(objectModel{objectID}.XYZ(2,:),T)&...
%           objectModel{objectID}.XYZ(2,:)<prctile(objectModel{objectID}.XYZ(2,:),100-T)&...
%           objectModel{objectID}.XYZ(3,:)>prctile(objectModel{objectID}.XYZ(3,:),T)&...
%           objectModel{objectID}.XYZ(3,:)<prctile(objectModel{objectID}.XYZ(3,:),100-T);
% objectModel{objectID}.XYZ =  objectModel{objectID}.XYZ(:,find(inrange));
% 
% % zero mean the model
% center = mean(objectModel{objectID}.XYZ,2);
% objectModel{objectID}.XYZ = bsxfun(@minus,objectModel{objectID}.XYZ,center);
% objectModel{objectID}.sift.P3D = bsxfun(@minus,objectModel{objectID}.sift.P3D,center);
% objectModel{objectID}.name = objnames{objectID};
% obj = objectModel{objectID};
% save(fullfile(datapath,objnames{objectID},'obj.mat'),'obj')
