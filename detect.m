function [objRt, detConf] = detect()

% (Temporary) System call to Period executable
system('export LD_LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/4.9; ./test')

imagePath = 'TMP.frame.color.png';
depthPath = 'TMP.frame.depth.png';
intrinsicsPath = 'TMP.intrinsics.K.txt';
resultsPath = 'TMP.results.txt';

% Sliding search cube dimensions (meters)
cubeDim = 0.15; 

% Load image and depth
I = imread(imagePath);
D = double(imread(depthPath))./1000;
K = dlmread(intrinsicsPath);
    
% Only load valid depth
D(find(D < 0.2)) = 0;
D(find(D > 0.8)) = 0;

% Get XYZ camera coordinates
[pixX,pixY] = meshgrid(1:640,1:480);
camX = (pixX-K(1,3)).*D/K(1,1); 
camY = (pixY-K(2,3)).*D/K(2,2); 
camZ = D;
camXYZ = [camX(:) camY(:) camZ(:)]';

% Load only valid XYZ points
validIDX = find(camXYZ(3,:) > 0);
camXYZ = camXYZ(:,validIDX);

% Get point colors
colorsR = I(:,:,1);
colorsG = I(:,:,2);
colorsB = I(:,:,3);
colors = cat(1,colorsR(:)',colorsG(:)',colorsB(:)');
colors = colors(:,validIDX);

% Load detected object location
objData = dlmread(resultsPath)';
detConf = objData(1);
objLoc = objData(2:4);
objBbox = [objLoc - cubeDim/2 objLoc + cubeDim/2];

% Get all 3D points within sampled cube
camIDX = intersect(intersect(intersect(find(camXYZ(1,:)>objBbox(1,1)),find(camXYZ(1,:)<objBbox(1,2))), ...
                             intersect(find(camXYZ(2,:)>objBbox(2,1)),find(camXYZ(2,:)<objBbox(2,2)))), ...
                             intersect(find(camXYZ(3,:)>objBbox(3,1)),find(camXYZ(3,:)<objBbox(3,2))));

% (Temporary) Use PCA of cropped 3D region as pose
pcaXYZ = pca(camXYZ(:,camIDX)');
pcaXYZ = pcaXYZ.*(1/10);
axisLoc = [objLoc, objLoc + pcaXYZ(:,1), objLoc + pcaXYZ(:,2), objLoc + pcaXYZ(:,3)];
axisLoc2D = axisLoc(1:2,:).*repmat([K(1,1);K(2,2)],1,4)./repmat(axisLoc(3,:),2,1)+repmat([K(1,3);K(2,3)],1,4);
imshow(I); 
hold on; plot(axisLoc2D(1,[1 2])',axisLoc2D(2,[1 2])','b','LineWidth',3); hold off;
hold on; plot(axisLoc2D(1,[1 3])',axisLoc2D(2,[1 3])','g','LineWidth',3); hold off;
hold on; plot(axisLoc2D(1,[1 4])',axisLoc2D(2,[1 4])','r','LineWidth',3); hold off;
eul = [atan2(norm(cross([1 0 0],pcaXYZ(:,3))),dot([1 0 0],pcaXYZ(:,3))), ...
       atan2(norm(cross([0 1 0],pcaXYZ(:,2))),dot([0 1 0],pcaXYZ(:,2))), ...
       atan2(norm(cross([0 0 1],pcaXYZ(:,1))),dot([0 0 1],pcaXYZ(:,1)))];
objRot = eul2rotm(eul);
objRt = [objRot, objLoc];

% % Show point cloud
% ptCloud = pointCloud(camXYZ(:,camIDX)','Color',colors(:,camIDX)');
% pcwrite(ptCloud,'cloud','PLYFormat','binary');
         