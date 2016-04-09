function label()
close all;
seqDir = '/home/mcube/software/period/data/train/elmers_washable_no_run_school_glue/000004';

%% Load all RGB-D frames and create a point cloud

% List RGB-D frames
colorFiles = dir(fullfile(seqDir,'*.color.png'));
depthFiles = dir(fullfile(seqDir,'*.depth.png'));

% Load intrinsics
K = dlmread(fullfile(seqDir,'intrinsics.K.txt'));

% Use extrinsics to create point cloud of sequence
objCoords = [];
objColors = [];
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
    
%     ptCloud = pointCloud(camXYZ','Color',colors');
%     pcwrite(ptCloud,sprintf('%d',frameIDX),'PLYFormat','binary');
    
    % Save points and colors
    objCoords = [objCoords,camXYZ];
    objColors = [objColors,colors];
end

% Create downsampled point cloud
ptCloud = pointCloud(objCoords','Color',objColors');
ptCloud = pcdownsample(ptCloud,'random',0.5);
% pcwrite(ptCloud,'test','PLYFormat','binary');


%% Create GUI to label data
f = figure; showPointCloud(ptCloud);

global poseLoc;
global poseRot;
poseLoc = [mean(ptCloud.XLimits)-0.2,mean(ptCloud.YLimits),mean(ptCloud.ZLimits)];
poseRot = eye(3);
axisR = [poseLoc - 0.05*poseRot(:,1)'; poseLoc + 0.25*poseRot(:,1)'];
axisG = [poseLoc - 0.05*poseRot(:,2)'; poseLoc + 0.25*poseRot(:,2)'];
axisB = [poseLoc - 0.05*poseRot(:,3)'; poseLoc + 0.25*poseRot(:,3)'];
hold on; hAxisR = plot3(axisR(:,1),axisR(:,2),axisR(:,3),'LineWidth',3,'Color','r'); hold off;
hold on; hAxisG = plot3(axisG(:,1),axisG(:,2),axisG(:,3),'LineWidth',3,'Color','g'); hold off;
hold on; hAxisB = plot3(axisB(:,1),axisB(:,2),axisB(:,3),'LineWidth',3,'Color','b'); hold off;

% GUI sliders for changing pose location
poseLocX = poseLoc(1);
poseLocY = poseLoc(2);
poseLocZ = poseLoc(3);
poseLocXcontrol = uicontrol('Parent',f,'Style','slider','Position',[81,200,1000,23],...
              'value',poseLocX, 'min',ptCloud.XLimits(1), 'max',ptCloud.XLimits(2));
poseLocXcontrol.Callback = @(es,ed) updateAxisR(hAxisR, hAxisG, hAxisB, es.Value); 
poseLocYcontrol = uicontrol('Parent',f,'Style','slider','Position',[81,170,1000,23],...
              'value',poseLocY, 'min',ptCloud.YLimits(1), 'max',ptCloud.YLimits(2));
poseLocYcontrol.Callback = @(es,ed) updateAxisG(hAxisR, hAxisG, hAxisB, es.Value); 
poseLocZcontrol = uicontrol('Parent',f,'Style','slider','Position',[81,140,1000,23],...
              'value',poseLocZ, 'min',ptCloud.ZLimits(1), 'max',ptCloud.ZLimits(2));
poseLocZcontrol.Callback = @(es,ed) updateAxisB(hAxisR, hAxisG, hAxisB, es.Value); 

% GUI sliders for changing pose rotation
eulX = 0;
eulY = 0;
eulZ = 0;
poseRotXcontrol = uicontrol('Parent',f,'Style','slider','Position',[81,110,1000,23],...
              'value',eulX, 'min',-pi, 'max',pi);
poseRotXcontrol.Callback = @(es,ed) updateAxisEulX(hAxisR, hAxisG, hAxisB, es.Value); 
poseRotYcontrol = uicontrol('Parent',f,'Style','slider','Position',[81,80,1000,23],...
              'value',eulY, 'min',-pi, 'max',pi);
poseRotYcontrol.Callback = @(es,ed) updateAxisEulY(hAxisR, hAxisG, hAxisB, es.Value); 
poseRotZcontrol = uicontrol('Parent',f,'Style','slider','Position',[81,50,1000,23],...
              'value',eulZ, 'min',-pi, 'max',pi);
poseRotZcontrol.Callback = @(es,ed) updateAxisEulZ(hAxisR, hAxisG, hAxisB, es.Value); 

% GUI button for saving object pose
saveBtn = uicontrol('Style', 'pushbutton', 'String', 'Save Object Pose',...
        'Position', [81 20 1000 20]); 
saveBtn.Callback = @(es,ed) saveObjPose(seqDir); 

end

function saveObjPose(seqDir)
    global poseLoc;
    global poseRot;
    Rt = cat(1,[poseRot,poseLoc'],[0 0 0 1]);
    dlmwrite(fullfile(seqDir,'object.pose.txt'),Rt,' ');
    close all;
    fprintf('Done.\n');
end

%% Redrawing functions
function redrawAxis(hAxisR, hAxisG, hAxisB)
    global poseLoc;
    global poseRot;
    axisR = [poseLoc - 0.1*poseRot(:,1)'; poseLoc + 0.25*poseRot(:,1)'];
    axisG = [poseLoc - 0.1*poseRot(:,2)'; poseLoc + 0.25*poseRot(:,2)'];
    axisB = [poseLoc - 0.1*poseRot(:,3)'; poseLoc + 0.25*poseRot(:,3)'];
    hAxisR.XData = axisR(:,1);
    hAxisR.YData = axisR(:,2);
    hAxisR.ZData = axisR(:,3);
    hAxisG.XData = axisG(:,1);
    hAxisG.YData = axisG(:,2);
    hAxisG.ZData = axisG(:,3);
    hAxisB.XData = axisB(:,1);
    hAxisB.YData = axisB(:,2);
    hAxisB.ZData = axisB(:,3);
end

function updateAxisEulX(hAxisR, hAxisG, hAxisB, eulX)
    global poseRot;
    poseEul = rot2eul(poseRot);
    poseEul(1) = eulX;
    poseRot = eul2rot(poseEul);
    redrawAxis(hAxisR, hAxisG, hAxisB);
end

function updateAxisEulY(hAxisR, hAxisG, hAxisB, eulY)
    global poseRot;
    poseEul = rot2eul(poseRot);
    poseEul(2) = eulY;
    poseRot = eul2rot(poseEul);
    redrawAxis(hAxisR, hAxisG, hAxisB);
end


function updateAxisEulZ(hAxisR, hAxisG, hAxisB, eulZ)
    global poseRot;
    poseEul = rot2eul(poseRot);
    poseEul(3) = eulZ;
    poseRot = eul2rot(poseEul);
    redrawAxis(hAxisR, hAxisG, hAxisB);
end

function updateAxisR(hAxisR, hAxisG, hAxisB, x)
    global poseLoc;
    poseLoc(1) = x;
    redrawAxis(hAxisR, hAxisG, hAxisB);
end


function updateAxisG(hAxisR, hAxisG, hAxisB, y)
    global poseLoc;
    poseLoc(2) = y;
    redrawAxis(hAxisR, hAxisG, hAxisB);
end

function updateAxisB(hAxisR, hAxisG, hAxisB, z)
    global poseLoc;
    poseLoc(3) = z;
    redrawAxis(hAxisR, hAxisG, hAxisB);
end








