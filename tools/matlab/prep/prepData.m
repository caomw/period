function prepData(n)

srcDir = 'data/frames/test';
dstDir = 'data/processed/test';

mkdir(dstDir);

numDataPoints = 1000;

% Sliding search cube dimensions (meters)
cubeDim = 0.15; 

% List RGB-D frames
colorFiles = dir(fullfile(srcDir,'*.color.png'));
depthFiles = dir(fullfile(srcDir,'*.depth.png'));

% Load intrinsics
K = dlmread(fullfile(srcDir,'intrinsics.K.txt'));

% Change random seed
rng(sum(100*clock),'twister');

dataCreated = 0;
while dataCreated < numDataPoints
    fprintf('%d/%d\n',dataCreated,numDataPoints);
    randFrameIDX = randsample(1:length(depthFiles),1);
    
    % Load RGB-D frame
    I = imread(fullfile(srcDir,colorFiles(randFrameIDX).name));
    D = double(imread(fullfile(srcDir,depthFiles(randFrameIDX).name)))./1000;
    
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
    
    % Get extrinsics
    frameName = colorFiles(randFrameIDX).name(1:(end-10));
    ext = dlmread(fullfile(srcDir,sprintf('%s.pose.txt',frameName)));
%     camXYZ = ext(1:3,1:3)*camXYZ + repmat(ext(1:3,4),1,size(camXYZ,2));
    
    % Get point colors
    colorsR = I(:,:,1);
    colorsG = I(:,:,2);
    colorsB = I(:,:,3);
    colors = cat(1,colorsR(:)',colorsG(:)',colorsB(:)');
    colors = colors(:,validIDX);
    
    % Load ground truth object bbox
    load('bbox.mat');
    
    % Convert bbox from world to camera coords
    bbox = ext(1:3,1:3)' * (bbox - repmat(ext(1:3,4),1,size(bbox,2)));
    gtCubeLoc = sum(bbox,2)/2;
    gtCubeLim = [gtCubeLoc-0.5*cubeDim,gtCubeLoc+0.5*cubeDim];
    
    % Compute view frustum
    viewFrust = [min(camXYZ,[],2),max(camXYZ,[],2)];
    viewFrust(3,1) = max(0.2,viewFrust(3,1));
    
    % Create data point hash name
    randASCII = [(48:57),(97:122)]; % caps: (65:90)
    hashASCII = randsample(randASCII,16,true);
    hashName = char(hashASCII);
    
    % Randomly select cubes until we have a positive and negative cube
    % Positive cube: within distance of ground truth cube
    % Negative cube: outside distance of ground truth cube
    posFound = false;
    negFound = false;
    iter = 0;
    while ~posFound || ~negFound
        iter = iter + 1;
        if iter > 100000 % Max iterations per image
            break;
        end
        randCubeLoc = (viewFrust(:,2)-viewFrust(:,1)).*rand(3,1) + viewFrust(:,1);
        
        % Check if cube is valid: 2D projection of cube is visible
        randCube = [randCubeLoc+[ 0.5*cubeDim; 0.5*cubeDim;-0.5*cubeDim], ...
                    randCubeLoc+[ 0.5*cubeDim;-0.5*cubeDim;-0.5*cubeDim], ...
                    randCubeLoc+[-0.5*cubeDim;-0.5*cubeDim;-0.5*cubeDim], ...
                    randCubeLoc+[-0.5*cubeDim; 0.5*cubeDim;-0.5*cubeDim]];
        randCube2D = round((randCube(1:2,:).*repmat([K(1,1);K(2,2)],1,size(randCube,2)))./repmat(randCube(3,:),2,1)+repmat([K(1,3);K(2,3)],1,size(randCube,2)));
        randCube2D = [randCube2D,randCube2D(:,1)];
        if min(randCube2D(1,:),[],2) < 1 || max(randCube2D(1,:),[],2) > 640 || ...
           min(randCube2D(2,:),[],2) < 1 || max(randCube2D(2,:),[],2) > 480  
            continue;
        end 
        
        % Compute 3D limits of sampled cube
        randCubeLim = [randCubeLoc-repmat(0.5*cubeDim,3,1),randCubeLoc+repmat(0.5*cubeDim,3,1)];
        
        % Compute distance to ground truth cube
        overlap = 0;
        if abs(gtCubeLim(1,1)-randCubeLim(1,1)) < cubeDim && ...
           abs(gtCubeLim(2,1)-randCubeLim(2,1)) < cubeDim && ...
           abs(gtCubeLim(3,1)-randCubeLim(3,1)) < cubeDim
            overlapX = min(abs(gtCubeLim(1,1)-randCubeLim(1,2)),abs(gtCubeLim(1,2)-randCubeLim(1,1)));
            overlapY = min(abs(gtCubeLim(2,1)-randCubeLim(2,2)),abs(gtCubeLim(2,2)-randCubeLim(2,1)));
            overlapZ = min(abs(gtCubeLim(3,1)-randCubeLim(3,2)),abs(gtCubeLim(3,2)-randCubeLim(3,1)));
            cubeIntersect = overlapX*overlapY*overlapZ;
            cubeUnion = cubeDim*cubeDim*cubeDim+cubeDim*cubeDim*cubeDim-cubeIntersect;
            overlap = cubeIntersect/cubeUnion;
        end
%         dist = sqrt(sum((randCubeLoc-gtCubeLoc).^2));   
        
        % Get all 3D points within sampled cube
        camIDX = intersect(intersect(intersect(find(camXYZ(1,:)>randCubeLim(1,1)),find(camXYZ(1,:)<randCubeLim(1,2))), ...
                                     intersect(find(camXYZ(2,:)>randCubeLim(2,1)),find(camXYZ(2,:)<randCubeLim(2,2)))), ...
                                     intersect(find(camXYZ(3,:)>randCubeLim(3,1)),find(camXYZ(3,:)<randCubeLim(3,2))));

        % Use only non-empty cubes
        if length(camIDX) < 100
            continue;
        end
                
        if overlap > 0.33; % Positive case
            if ~posFound
%                 randCubeLim = [randCubeLoc-repmat(0.5*cubeDim,3,1),randCubeLoc+repmat(0.5*cubeDim,3,1)];
%                 camIDX = intersect(intersect(intersect(find(camXYZ(1,:)>randCubeLim(1,1)),find(camXYZ(1,:)<randCubeLim(1,2))), ...
%                                              intersect(find(camXYZ(2,:)>randCubeLim(2,1)),find(camXYZ(2,:)<randCubeLim(2,2)))), ...
%                                              intersect(find(camXYZ(3,:)>randCubeLim(3,1)),find(camXYZ(3,:)<randCubeLim(3,2))));
%                 
%                 % Use only non-empty cubes
%                 if length(camIDX) < 100
%                     continue;
%                 end
                             
                % Save RGB-D patch and point cloud                
                patchRGB = I(min(randCube2D(2,:)):max(randCube2D(2,:)),min(randCube2D(1,:)):max(randCube2D(1,:)),:);
                imwrite(patchRGB,fullfile(dstDir,sprintf('pos.%s.color.png',hashName)));
                patchD = uint16(D(min(randCube2D(2,:)):max(randCube2D(2,:)),min(randCube2D(1,:)):max(randCube2D(1,:)),:)*1000);
                imwrite(patchD,fullfile(dstDir,sprintf('pos.%s.depth.png',hashName)));
%                 ptCloud = pointCloud(camXYZ(:,camIDX)','Color',colors(:,camIDX)');
%                 pcwrite(ptCloud,fullfile(dstDir,sprintf('pos.%s.cloud',hashName)),'PLYFormat','binary');

                % Convert point cloud to 30x30x30 tsdf
                tsdf = ones(30,30,30,3);
                tsdfPts = []; tsdfColors = [];
                trunc = cubeDim/30;
                [gridX,gridY,gridZ] = meshgrid(1:30,1:30,1:30);
                grid = [gridX(:) gridY(:) gridZ(:)]';
                voxLim = cat(3,repmat(randCubeLim(:,1),1,size(grid,2))+(grid-1)*cubeDim/30,repmat(randCubeLim(:,1),1,size(grid,2))+grid*cubeDim/30);
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
%                 ptCloud = pointCloud(tsdfPts','Color',uint8(tsdfColors'));
%                 pcwrite(ptCloud,fullfile(dstDir,sprintf('pos.%s.tsdf',hashName)),'PLYFormat','binary');
                save(fullfile(dstDir,sprintf('pos.%s.tsdf.mat',hashName)),'tsdf');
                
                posFound = true;
            end
        else % Negative case
            if ~negFound
                
                % Save RGB-D patch and point cloud                
                patchRGB = I(min(randCube2D(2,:)):max(randCube2D(2,:)),min(randCube2D(1,:)):max(randCube2D(1,:)),:);
                imwrite(patchRGB,fullfile(dstDir,sprintf('neg.%s.color.png',hashName)));
                patchD = uint16(D(min(randCube2D(2,:)):max(randCube2D(2,:)),min(randCube2D(1,:)):max(randCube2D(1,:)),:)*1000);
                imwrite(patchD,fullfile(dstDir,sprintf('neg.%s.depth.png',hashName)));
%                 ptCloud = pointCloud(camXYZ(:,camIDX)','Color',colors(:,camIDX)');
%                 pcwrite(ptCloud,fullfile(dstDir,sprintf('neg.%s.cloud',hashName)),'PLYFormat','binary');

                % Convert point cloud to 30x30x30 tsdf
                tsdf = ones(30,30,30,3);
                tsdfPts = []; tsdfColors = [];
                trunc = cubeDim/30;
                [gridX,gridY,gridZ] = meshgrid(1:30,1:30,1:30);
                grid = [gridX(:) gridY(:) gridZ(:)]';
                voxLim = cat(3,repmat(randCubeLim(:,1),1,size(grid,2))+(grid-1)*cubeDim/30,repmat(randCubeLim(:,1),1,size(grid,2))+grid*cubeDim/30);
                voxLoc = (voxLim(:,:,2)+voxLim(:,:,1))/2;
                for i = 1:size(voxLoc,2)
                    tmpLoc = voxLoc(:,i);
                    [dist,minIDX] = min(sqrt(sum((repmat(tmpLoc,1,size(camXYZ,2))-camXYZ).^2)));
%                     if tmpLoc(3) > camXYZ(3,minIDX) % Negative sign
%                         dist = -dist;
%                     end
%                     tsdf(grid(1,i),grid(2,i),grid(3,i)) = max(-1,min(1,dist/trunc));
                    dirSDF = camXYZ(:,minIDX)-tmpLoc;
                    dirSDF = max([[-1;-1;-1],min([[1;1;1],dirSDF/trunc],[],2)],[],2);
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
%                 ptCloud = pointCloud(tsdfPts','Color',uint8(tsdfColors'));
%                 pcwrite(ptCloud,fullfile(dstDir,sprintf('neg.%s.tsdf',hashName)),'PLYFormat','binary');
                save(fullfile(dstDir,sprintf('neg.%s.tsdf.mat',hashName)),'tsdf');
                
                negFound = true;
            end
        end
    end
    
    if posFound && negFound
        dataCreated = dataCreated + 1;
    end
end







