
objCoords = [];
objColors = [];

seqDir = 'data/glue.train.5';

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
    try
        ext = dlmread(fullfile(seqDir,sprintf('%s.pose.txt',frameName)));
    catch err
        break;
    end
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
fprintf('Open the "object.ply" file in Meshlab and remove all non-object points.\nPress any key to continue...\n');
pause;

% Load point cloud
ptCloud = pcread(sprintf('%s.ply','object'));

% Get object's 3D bbox from point cloud
bbox = [ptCloud.XLimits; ptCloud.YLimits; ptCloud.ZLimits];
objLoc = [sum(ptCloud.XLimits)/2 sum(ptCloud.YLimits)/2 sum(ptCloud.ZLimits)/2]';

% delete('object.ply');
% dlmwrite(fullfile(seqDir,'object.bbox.txt'),bbox,' ');
% save('object.bbox.txt','bbox');

% Compute PCA of object
coeff = pca(ptCloud.Location);
coeff = coeff./10;
axisPCA = [(objLoc-coeff(:,1)),(objLoc+coeff(:,1)),(objLoc-coeff(:,2)),(objLoc+coeff(:,2)),(objLoc-coeff(:,3)),(objLoc+coeff(:,3))];
axisPCA = (axisPCA.*repmat([K(1,1);K(2,2);1],1,6)./repmat(axisPCA(3,:),3,1))+repmat([K(1,3);K(2,3);0],1,6);

% Draw object axes (PCs)
figure(); imshow(imread(fullfile(seqDir,colorFiles(1).name)));
hold on;
plot(axisPCA(1,1:2)',axisPCA(2,1:2)','y','LineWidth',3);
scatter(axisPCA(1,1)',axisPCA(2,1)',100,'y','fill');
plot(axisPCA(1,3:4)',axisPCA(2,3:4)','m','LineWidth',3);
scatter(axisPCA(1,3)',axisPCA(2,3)',100,'m','fill');
plot(axisPCA(1,5:6)',axisPCA(2,5:6)','c','LineWidth',3);
scatter(axisPCA(1,5)',axisPCA(2,5)',100,'c','fill');
hold off

% Verify axes of object
str = input('Flip the yellow line (y/n)? ','s');
if strcmp(str,'y')
    coeff(:,1) = -coeff(:,1);
end
str = input('Flip the magenta line (y/n)? ','s');
if strcmp(str,'y')
    coeff(:,2) = -coeff(:,2);
end
str = input('Flip the cyan line (y/n)? ','s');
if strcmp(str,'y')
    coeff(:,3) = -coeff(:,3);
end
objAxes = coeff;
str = input('What color should the yellow line be (r/g/b)? ','s');
switch str
    case 'r'
        hold on; plot(axisPCA(1,1:2)',axisPCA(2,1:2)','r','LineWidth',3);
        scatter(axisPCA(1,1)',axisPCA(2,1)',100,'r','fill'); hold off;
        objAxes(:,1) = coeff(:,1);
    case 'g'
        hold on; plot(axisPCA(1,1:2)',axisPCA(2,1:2)','g','LineWidth',3);
        scatter(axisPCA(1,1)',axisPCA(2,1)',100,'g','fill'); hold off;
        objAxes(:,2) = coeff(:,1);
    case 'b'
        hold on; plot(axisPCA(1,1:2)',axisPCA(2,1:2)','b','LineWidth',3);
        scatter(axisPCA(1,1)',axisPCA(2,1)',100,'b','fill'); hold off;
        objAxes(:,3) = coeff(:,1);
end
str = input('What color should the magenta line be (r/g/b)? ','s');
switch str
    case 'r'
        hold on; plot(axisPCA(1,3:4)',axisPCA(2,3:4)','r','LineWidth',3);
        scatter(axisPCA(1,3)',axisPCA(2,3)',100,'r','fill'); hold off;
        objAxes(:,1) = coeff(:,2);
    case 'g'
        hold on; plot(axisPCA(1,3:4)',axisPCA(2,3:4)','g','LineWidth',3);
        scatter(axisPCA(1,3)',axisPCA(2,3)',100,'g','fill'); hold off;
        objAxes(:,2) = coeff(:,2);
    case 'b'
        hold on; plot(axisPCA(1,3:4)',axisPCA(2,3:4)','b','LineWidth',3);
        scatter(axisPCA(1,3)',axisPCA(2,3)',100,'b','fill'); hold off;
        objAxes(:,3) = coeff(:,2);
end
str = input('What color should the cyan line be (r/g/b)? ','s');
switch str
    case 'r'
        hold on; plot(axisPCA(1,5:6)',axisPCA(2,5:6)','r','LineWidth',3);
        scatter(axisPCA(1,5)',axisPCA(2,5)',100,'r','fill'); hold off;
        objAxes(:,1) = coeff(:,3);
    case 'g'
        hold on; plot(axisPCA(1,5:6)',axisPCA(2,5:6)','g','LineWidth',3);
        scatter(axisPCA(1,5)',axisPCA(2,5)',100,'g','fill'); hold off;
        objAxes(:,2) = coeff(:,3);
    case 'b'
        hold on; plot(axisPCA(1,5:6)',axisPCA(2,5:6)','b','LineWidth',3);
        scatter(axisPCA(1,5)',axisPCA(2,5)',100,'b','fill'); hold off;
        objAxes(:,3) = coeff(:,3);
end

% Re-draw object axes
coeff = objAxes;
axisPCA = [(objLoc-coeff(:,1)),(objLoc+coeff(:,1)),(objLoc-coeff(:,2)),(objLoc+coeff(:,2)),(objLoc-coeff(:,3)),(objLoc+coeff(:,3))];
axisPCA = (axisPCA.*repmat([K(1,1);K(2,2);1],1,6)./repmat(axisPCA(3,:),3,1))+repmat([K(1,3);K(2,3);0],1,6);
imshow(imread(fullfile(seqDir,colorFiles(1).name)));
hold on;
plot(axisPCA(1,1:2)',axisPCA(2,1:2)','r','LineWidth',3);
scatter(axisPCA(1,1)',axisPCA(2,1)',100,'r','fill');
plot(axisPCA(1,3:4)',axisPCA(2,3:4)','g','LineWidth',3);
scatter(axisPCA(1,3)',axisPCA(2,3)',100,'g','fill');
plot(axisPCA(1,5:6)',axisPCA(2,5:6)','b','LineWidth',3);
scatter(axisPCA(1,5)',axisPCA(2,5)',100,'b','fill');
hold off

% Compute rigid transform
camAxis = eye(3);
coeff = coeff.*10;
rot = coeff/camAxis;
Rt = cat(1,[rot,objLoc],[0 0 0 1]);

% Rotation matrix to Euler axis/angle
eAxisAngle = vrrotmat2vec(rot);
rotm = vrrotvec2mat(eAxisAngle);
% eAxis = eAxisAngle(1:3);
% eAngle = eAxisAngle(4);
rot
rotm

% Save object location and orientation
dlmwrite(fullfile(seqDir,'object.pose.txt'),Rt,' ');

% % Find closest point to y-axis component on unit sphere
% spherePts = icosahedron2sphere(2);
% dist = sqrt(sum((spherePts-repmat(eAxis,size(spherePts,1),1)).^2,2));
% [minVal,minIdx] = min(dist);
% closestPt = spherePts(minIdx,:)';












