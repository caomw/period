%% Multi-camera calibration
% Using two RGB frame sequences (captured from two difference cameras) of
% a chessboard, estimate the rigid transform between the cameras

% --------------- Changeable parameters --------------------
% Specify checkerboard square sizes in units of 'mm'
checkerBoardSquareSize = 30;

% Specify file locations
dataDir = '../../../data';
calibSeq = fullfile(dataDir,'calib12'); % where to read chessboard calibration frames

% Get filenames of first frame sequence 
seq1Color = dir(fullfile(calibSeq,'*.0.color.png'));
seq1Depth = dir(fullfile(calibSeq,'*.0.depth.png'));

% Get filenames of second frame sequence 
seq2Color = dir(fullfile(calibSeq,'*.1.color.png'));
seq2Depth = dir(fullfile(calibSeq,'*.1.depth.png'));

% Load intrinsics
K1 = dlmread(fullfile(calibSeq,'intrinsics.0.K.txt'));
K2 = dlmread(fullfile(calibSeq,'intrinsics.1.K.txt'));

% Get destination filenames
dstExtFilename = fullfile(dataDir,'extrinsics.2.pose.txt'); % where to save inter-camera rigid transform
dstInvertExt = false;

% ----------------------------------------------------------

seq1ColorFilenames = {};
seq1DepthFilenames = {};
seq2ColorFilenames = {};
seq2DepthFilenames = {};
for frameIDX = 1:min(length(seq1Color),length(seq2Color))
    seq1ColorFilenames{length(seq1ColorFilenames) + 1} = fullfile(calibSeq,seq1Color(frameIDX).name);
    seq1DepthFilenames{length(seq1DepthFilenames) + 1} = fullfile(calibSeq,seq1Depth(frameIDX).name);
    seq2ColorFilenames{length(seq2ColorFilenames) + 1} = fullfile(calibSeq,seq2Color(frameIDX).name);
    seq2DepthFilenames{length(seq2DepthFilenames) + 1} = fullfile(calibSeq,seq2Depth(frameIDX).name);
end
fprintf('Calibrating...\n');

% Detect checkerboards in images
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(seq1ColorFilenames, seq2ColorFilenames);

% Generate camera coordinates of the checkerboard keypoints
frameIDX = find(imagesUsed > 0);
camPoints1 = []; camPoints2 = [];
for i=1:size(imagePoints,3)

    % Get depth frames
    D1 = double(imread(seq1DepthFilenames{frameIDX(i)}))./1000;
    D1 = propagateDepth(D1);
    D2 = double(imread(seq2DepthFilenames{frameIDX(i)}))./1000;
    D2 = propagateDepth(D2);
    
    % Only get valid depth values
    D1(find(D1 < 0.2)) = 0; D1(find(D1 > 0.8)) = 0;
    D2(find(D2 < 0.2)) = 0; D2(find(D2 > 0.8)) = 0;
    tmpCamPoints1 = round(imagePoints(:,:,i,1));
    tmpCamPoints2 = round(imagePoints(:,:,i,2));
    tmpCamPoints1 = [tmpCamPoints1, D1(sub2ind(size(D1),tmpCamPoints1(:,2),tmpCamPoints1(:,1)))];
    tmpCamPoints2 = [tmpCamPoints2, D2(sub2ind(size(D2),tmpCamPoints2(:,2),tmpCamPoints2(:,1)))];
    tmpCamPointsInd = intersect(find(tmpCamPoints1(:,3)>0),find(tmpCamPoints2(:,3)>0));
    tmpCamPoints1 = tmpCamPoints1(tmpCamPointsInd,:);
    tmpCamPoints2 = tmpCamPoints2(tmpCamPointsInd,:);
    
    % Project points to camera space
    tmpCamPoints1(:,1) = (tmpCamPoints1(:,1)-K1(1,3)).*tmpCamPoints1(:,3)/K1(1,1); 
    tmpCamPoints1(:,2) = (tmpCamPoints1(:,2)-K1(2,3)).*tmpCamPoints1(:,3)/K1(2,2); 
    tmpCamPoints2(:,1) = (tmpCamPoints2(:,1)-K2(1,3)).*tmpCamPoints2(:,3)/K2(1,1); 
    tmpCamPoints2(:,2) = (tmpCamPoints2(:,2)-K2(2,3)).*tmpCamPoints2(:,3)/K2(2,2); 
    
    camPoints1 = [camPoints1; tmpCamPoints1];
    camPoints2 = [camPoints2; tmpCamPoints2];

end

% Compute rigid transform using least-squares
addpath(fullfile('lib/estimateRigidTransform'));
addpath(fullfile('lib/peter'));
[RT, inliers] = ransacfitRt([camPoints1'; camPoints2'], 0.01, 0);
RT = [RT; 0 0 0 1];
% [RT, Eps] = estimateRigidTransform(camPoints1', camPoints2');

if dstInvertExt
    RT = inv(RT);
end

% Save rigid transform from second camera to first camera
fileID = fopen(dstExtFilename,'w');
for i = 1:4
    fprintf(fileID, '%.17g %.17g %.17g %.17g\n',RT(i,1),RT(i,2),RT(i,3),RT(i,4));
end
fclose(fileID);