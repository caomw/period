


seqDir = '../../../data/train/book/seq01';
numTags = 4;

% List RGB-D frames
colorFiles = dir(fullfile(seqDir,'*.color.png'));
apriltagFiles = dir(fullfile(seqDir,'*.apriltags.txt'));
numFrames = length(colorFiles);

allPts = [];
allColors = [];
for frameIDX=1:numFrames
    fprintf('Loading frame %d/%d\n',frameIDX,length(depthFiles));
    
    % Load RGB-D frame
    I = imread(fullfile(seqDir,colorFiles(frameIDX).name));
    D = double(imread(fullfile(seqDir,depthFiles(frameIDX).name)))./1000;
    
    % Set invalid depth to 0
    D(find(D < 0.2)) = 0;
    D(find(D > 0.8)) = 0;
    
    % Fill holes in depth using local max
    D = fillHolesDepth(D);
    
    % Get frame name prefix
    frameName = colorFiles(frameIDX).name(1:(end-10));
    
    % Compute extrinsics from main tag
    if ~isempty(cam2tag{frameIDX,mainTagIDX})
        extrinsics{frameIDX} = cam2tag{frameIDX,mainTagIDX};
    else
        for tagIDX = 1:numTags
            if ~isempty(cam2tag{frameIDX,tagIDX})
                cam2alt = cam2tag{frameIDX,tagIDX};
                cam2main = alt2mainTagRt{tagIDX} * cam2alt;
                extrinsics{frameIDX} = cam2main;
                break;
            end
        end
    end
    
    % Get XYZ points in camera coordinate space
    [pixX,pixY] = meshgrid(1:640,1:480);
    camX = (pixX-K(1,3)).*D/K(1,1);
    camY = (pixY-K(2,3)).*D/K(2,2);
    camZ = D;
    camXYZ = cat(3,camX,camY,camZ,(camZ > 0));
    validIDX = camZ > 0;
    currFramePts = [camX(validIDX) camY(validIDX) camZ(validIDX)]';
    
    % Get point colors
    colorsR = I(:,:,1);
    colorsG = I(:,:,2);
    colorsB = I(:,:,3);
    currColors = cat(1,colorsR(:)',colorsG(:)',colorsB(:)');
    currColors = currColors(:,validIDX);
    
    currRt = extrinsics{frameIDX};
    currFramePts = currRt(1:3,1:3) * currFramePts + repmat(currRt(1:3,4),1,size(currFramePts,2));
    
    allPts = [allPts, currFramePts];
    allColors = [allColors, currColors];
    
end

ptCloud = pointCloud(allPts','Color',allColors');
ptCloud = pcdownsample(ptCloud,'random',0.1);
pcwrite(ptCloud,'pointcloud','PLYFormat','binary');


f = figure; pcshow(ptCloud);
zeta = 1;
b = uicontrol('Parent',f,'Style','slider','Position',[81,54,419,23],...
              'value',zeta, 'min',0, 'max',1);







