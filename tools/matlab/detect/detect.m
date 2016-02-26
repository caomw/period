clear all; close all;
addpath('../../marvin/tools/tensorIO_matlab');

imagePath = '../../../data/test1/frame-000008.color.png';
depthPath = '../../../data/test1/frame-000008.depth.png';
intrinsicsPath = '../../../data/test1/intrinsics.K.txt';

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

% Compute view frustum
viewFrust = [min(camXYZ,[],2),max(camXYZ,[],2)];
viewFrust(3,1) = max(0.2,viewFrust(3,1));

% Generate potential cube locations
[cubeLocX, cubeLocY, cubeLocZ] = meshgrid(viewFrust(1,1):0.02:viewFrust(1,2),...
                                          viewFrust(2,1):0.02:viewFrust(2,2),...
                                          viewFrust(3,1):0.02:viewFrust(3,2));
cubeLocs = [cubeLocX(:),cubeLocY(:),cubeLocZ(:)]';
validCubeLocs = zeros(1,size(cubeLocs,2));

% Find all cubes that are not empty
for cubeIDX = 1:size(cubeLocs,2)
    cubeIDX
    tmpCubeLoc = cubeLocs(:,cubeIDX);
        
    % Check if cube is valid: 2D projection of cube is visible
    tmpCube = [tmpCubeLoc+[ 0.5*cubeDim; 0.5*cubeDim;-0.5*cubeDim], ...
                tmpCubeLoc+[ 0.5*cubeDim;-0.5*cubeDim;-0.5*cubeDim], ...
                tmpCubeLoc+[-0.5*cubeDim;-0.5*cubeDim;-0.5*cubeDim], ...
                tmpCubeLoc+[-0.5*cubeDim; 0.5*cubeDim;-0.5*cubeDim]];
    tmpCube2D = round((tmpCube(1:2,:).*repmat([K(1,1);K(2,2)],1,size(tmpCube,2)))./repmat(tmpCube(3,:),2,1)+repmat([K(1,3);K(2,3)],1,size(tmpCube,2)));
    tmpCube2D = [tmpCube2D,tmpCube2D(:,1)];
    if min(tmpCube2D(1,:),[],2) < 1 || max(tmpCube2D(1,:),[],2) > 640 || ...
       min(tmpCube2D(2,:),[],2) < 1 || max(tmpCube2D(2,:),[],2) > 480  
        continue;
    end 

    % Compute 3D limits of sampled cube
    tmpCubeLim = [tmpCubeLoc-repmat(0.5*cubeDim,3,1),tmpCubeLoc+repmat(0.5*cubeDim,3,1)];

    % Get all 3D points within sampled cube
    camIDX = intersect(intersect(intersect(find(camXYZ(1,:)>tmpCubeLim(1,1)),find(camXYZ(1,:)<tmpCubeLim(1,2))), ...
                                 intersect(find(camXYZ(2,:)>tmpCubeLim(2,1)),find(camXYZ(2,:)<tmpCubeLim(2,2)))), ...
                                 intersect(find(camXYZ(3,:)>tmpCubeLim(3,1)),find(camXYZ(3,:)<tmpCubeLim(3,2))));

    % Use only non-empty cubes
    if length(camIDX) < 100
        continue;
    end
    
    % Save index of valid cube
    validCubeLocs(cubeIDX) = 1; 
    
    % For valid cubes,
    % Save RGB-D patch and point cloud                
    patchRGB = I(min(tmpCube2D(2,:)):max(tmpCube2D(2,:)),min(tmpCube2D(1,:)):max(tmpCube2D(1,:)),:);
%     imwrite(patchRGB,fullfile(dstDir,sprintf('pos.%s.color.png',hashName)));
%     patchD = uint16(D(min(randCube2D(2,:)):max(randCube2D(2,:)),min(randCube2D(1,:)):max(randCube2D(1,:)),:)*1000);
%     imwrite(patchD,fullfile(dstDir,sprintf('pos.%s.depth.png',hashName)));
% %                 ptCloud = pointCloud(camXYZ(:,camIDX)','Color',colors(:,camIDX)');
% %                 pcwrite(ptCloud,fullfile(dstDir,sprintf('pos.%s.cloud',hashName)),'PLYFormat','binary');

    % Convert point cloud to 30x30x30 tsdf
    tsdf = ones(30,30,30,3);
    tsdfPts = []; tsdfColors = [];
    trunc = cubeDim/30;
    [gridX,gridY,gridZ] = meshgrid(1:30,1:30,1:30);
    grid = [gridX(:) gridY(:) gridZ(:)]';
    voxLim = cat(3,repmat(tmpCubeLim(:,1),1,size(grid,2))+(grid-1)*cubeDim/30,repmat(tmpCubeLim(:,1),1,size(grid,2))+grid*cubeDim/30);
    voxLoc = (voxLim(:,:,2)+voxLim(:,:,1))/2;
    for i = 1:size(voxLoc,2)
        tmpLoc = voxLoc(:,i);
        [dist,minIDX] = min(sqrt(sum((repmat(tmpLoc,1,size(camXYZ,2))-camXYZ).^2)));
%                     if tmpLoc(3) > camXYZ(3,minIDX) % Negative sign
%                         dist = -dist;
%                     end
%                     tsdf(grid(1,i),grid(2,i),grid(3,i)) = max(-1,min(1,dist/trunc));
        dirSDF = camXYZ(:,minIDX)-tmpLoc;
        dirSDF = max([[-1;-1;-1],min([[1;1;1],dirSDF./trunc],[],2)],[],2);
        tsdf(grid(1,i),grid(2,i),grid(3,i),:) = reshape(dirSDF,1,1,1,3);
        if sum(abs(dirSDF) < 1) == 3
            tsdfPts = [tsdfPts,grid(:,i)];
            if dirSDF(3) < 0
                tsdfColors = [tsdfColors,[255;0;0]];
            else
                tsdfColors = [tsdfColors,[0;0;255]];
            end
        end
    end
    
    
    
end
cubeLocs = cubeLocs(:,find(validCubeLocs));


for cubeIDX = 1:size(cubeLocs,2)
    
    
    
end
