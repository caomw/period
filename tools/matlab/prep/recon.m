function recon(seqDir)
numTags = 4;

% List RGB-D frames
colorFiles = dir(fullfile(seqDir,'*.color.png'));
apriltagFiles = dir(fullfile(seqDir,'*.apriltags.txt'));
numFrames = length(colorFiles);

% Remove frames without apriltags
for frameIDX=1:numFrames
    try
        currFrameTagPoses = dlmread(fullfile(seqDir,apriltagFiles(frameIDX).name));
    catch
        frameName = colorFiles(frameIDX).name(1:(end-10));
        delete(fullfile(seqDir,[frameName,'*']));
    end
end

% % Add library paths
% addpath(fullfile('lib/icp'));

% Re-list RGB-D frames
colorFiles = dir(fullfile(seqDir,'*.color.png'));
depthFiles = dir(fullfile(seqDir,'*.depth.png'));
apriltagFiles = dir(fullfile(seqDir,'*.apriltags.txt'));
numFrames = length(depthFiles);

% Compute cam2tag transforms
cam2tag = cell(numFrames,numTags);
tagNumRt = zeros(1,numTags);
for frameIDX=1:numFrames
    currFrameTagPoses = dlmread(fullfile(seqDir,apriltagFiles(frameIDX).name));
    for i=1:(size(currFrameTagPoses,1)/5)
        tagIDX = currFrameTagPoses(i*5-4,1) + 1; % IDs start at 1 here rather than 0
        cam2tag{frameIDX,tagIDX} = inv(currFrameTagPoses((i*5-3):(i*5), 1:4));
        tagNumRt(tagIDX) = tagNumRt(tagIDX) + 1;
    end   
end

% Find which tag occurs most often and compute its Rt to other tags
mainTagIDX = find(tagNumRt == max(tagNumRt));
alt2mainTagRt = cell(1,numTags);
for altTagIDX = 1:numTags
    if altTagIDX == mainTagIDX
        continue;
    end
    
    % Get average Rt between tags
    numRt = 0;
    avgQuat = zeros(1,4);
    avgTrans = zeros(3,1);
    for frameIDX=1:numFrames
        if isempty(cam2tag{frameIDX,altTagIDX}) || isempty(cam2tag{frameIDX,mainTagIDX})
            continue;
        end
        alt2cam = inv(cam2tag{frameIDX,altTagIDX});
        cam2main = cam2tag{frameIDX,mainTagIDX};
        alt2main = cam2main * alt2cam;
        avgQuat = avgQuat + rot2quat(alt2main(1:3,1:3));
        avgTrans = avgTrans + alt2main(1:3,4);
        numRt = numRt + 1;
    end
    avgQuat = normr(avgQuat/numRt);
    avgTrans = avgTrans/numRt;
    
    % Convert average Rt to rotation matrix
    avgRot = eye(4);
    avgRot(1:3,1:3) = quat2rot(avgQuat);
    avgRot(1:3,4) = avgTrans;
    alt2mainTagRt{altTagIDX} = avgRot;
end

% Loop through each frame and compute extrinsics
extrinsics = cell(1,numFrames);
for frameIDX=1:numFrames
    
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
    currRt = extrinsics{frameIDX};
    
    % Get frame name prefix
    frameName = fullfile(seqDir,colorFiles(frameIDX).name);
    frameName = frameName(1:(end-10));
    
    % Save extrinsics
    fileID = fopen(strcat(frameName,'.pose.txt'),'w');
    for i = 1:4
        fprintf(fileID, '%.17g %.17g %.17g %.17g\n',currRt(i,1),currRt(i,2),currRt(i,3),currRt(i,4));
    end
    fclose(fileID);
end

end

%% BACKUP
%     if frameIDX == 1
%         
%         % Compute 3D bbox of object (specified by user)
%         fprintf('Select two points (upper left, bottom right) to mark the region of interest.\n');
%         figure(); imshow(I);
%         ptsMarked = 0;
%         while ptsMarked < 1
%             [gUpperLeftX,gUpperLeftY] = ginput(1);
%             if D(round(gUpperLeftY),round(gUpperLeftX)) > 0
%                 ptsMarked = ptsMarked + 1;
%                 hold on; scatter(gUpperLeftX,gUpperLeftY,'x'); hold off;
%                 fprintf('Point selected!\n');
%             else
%                 fprintf('Selected point has no depth. Please try again.\n');
%             end
%         end
%         while ptsMarked < 2
%             [gBottomRightX,gBottomRightY] = ginput(1);
%             if D(round(gBottomRightY),round(gBottomRightX)) > 0
%                 ptsMarked = ptsMarked + 1;
%                 hold on; scatter(gBottomRightX,gBottomRightY,'x'); hold off;
%                 fprintf('Point selected!\n');
%             else
%                 fprintf('Selected point has no depth. Please try again.\n');
%             end
%         end
%         camXYZpatch = camXYZ(gUpperLeftY:gBottomRightY,gUpperLeftX:gBottomRightX,1:3);
%         validPatchIDX = find(camXYZpatch(:,:,3) > 0);
%         camXpatch = camXYZpatch(:,:,1);
%         camXpatch = camXpatch(validPatchIDX);
%         camYpatch = camXYZpatch(:,:,2);
%         camYpatch = camYpatch(validPatchIDX);
%         camZpatch = camXYZpatch(:,:,3);
%         camZpatch = camZpatch(validPatchIDX);
%         camLimitsXYZ = [min(camXpatch) max(camXpatch); min(camYpatch) max(camYpatch); min(camZpatch) max(camZpatch)];
%         
%         % Prune points in 3D
%         validIndX = intersect(find(currFramePts(1,:) > camLimitsXYZ(1,1)),find(currFramePts(1,:) < camLimitsXYZ(1,2)));
%         validIndY = intersect(find(currFramePts(2,:) > camLimitsXYZ(2,1)),find(currFramePts(2,:) < camLimitsXYZ(2,2)));
%         validIndZ = intersect(find(currFramePts(3,:) > camLimitsXYZ(3,1)),find(currFramePts(3,:) < camLimitsXYZ(3,2)));
%         validIndXYZ = intersect(intersect(validIndX,validIndY),validIndZ);
%         currFramePts = currFramePts(:,validIndXYZ');
%         currColors = currColors(:,validIndXYZ');
%         
%         cloudPts{frameIDX} = currFramePts;
%         cloudColors{frameIDX} = currColors;
%         extrinsics{frameIDX} = eye(4);
%     else
%         
% %         % Find top k relative transforms closest to identity matrix
% %         rotm2eul
% %         rtDist = zeros(length(currFrameRelativeTransforms),1);
% %         for i=1:length(currFrameRelativeTransforms)
% %             rtDist(i) = sum(sum(sqrt((currFrameRelativeTransforms{i}(1:3,1:3) - eye(3)).^2)));
% %         end
% %         [sortedRtDist, sortIDX] = sortrows(rtDist,1);
% %         
% %         % Find mean relative transform from top k
% %         k = 10;
% %         meanRelativeTransform = zeros(4,4);
% %         for i=1:max(length(currFrameRelativeTransforms),k)
% %             meanRelativeTransform = meanRelativeTransform + currFrameRelativeTransforms{sortIDX(i)};
% %         end
% %         meanRelativeTransform = meanRelativeTransform./(max(length(currFrameRelativeTransforms),k));
% %         currRt = meanRelativeTransform;
% 
%         currRt = currFrameRelativeTransforms{1};
%         
%         % Prune points in 3D
%         invExtrinsics = inv(extrinsics{frameIDX - 1} * currRt);
%         currCamLimitsXYZ = invExtrinsics(1:3,1:3) * camLimitsXYZ + repmat(invExtrinsics(1:3,4),1,size(camLimitsXYZ,2));
%         validIndX = intersect(find(currFramePts(1,:) > currCamLimitsXYZ(1,1)),find(currFramePts(1,:) < currCamLimitsXYZ(1,2)));
%         validIndY = intersect(find(currFramePts(2,:) > currCamLimitsXYZ(2,1)),find(currFramePts(2,:) < currCamLimitsXYZ(2,2)));
%         validIndZ = intersect(find(currFramePts(3,:) > currCamLimitsXYZ(3,1)),find(currFramePts(3,:) < currCamLimitsXYZ(3,2)));
%         validIndXYZ = intersect(intersect(validIndX,validIndY),validIndZ);
%         currFramePts = currFramePts(:,validIndXYZ');
%         currColors = currColors(:,validIndXYZ');
%         
%         currTransformPts = currRt(1:3,1:3) * currFramePts + repmat(currRt(1:3,4),1,size(currFramePts,2));
%         prevTransformPts = cloudPts{frameIDX - 1};
%         currICPTransform = eye(4);
%         [TR, TT, ER, maxD, t] = icp(prevTransformPts,currTransformPts);
%         currICPTransform(1:3,1:3) = TR;
%         currICPTransform(1:3,4) = TT;
%         extrinsics{frameIDX} = extrinsics{frameIDX - 1} * currICPTransform * currRt;
%         cloudPts{frameIDX} = currFramePts;
%         cloudColors{frameIDX} = currColors;
%     end

%% BACKUP
% allPts = [];
% allColors = [];
% for frameIDX=1:20
%     currPts = cloudPts{frameIDX};
%     currExt = extrinsics{frameIDX};
%     currPts = currExt(1:3,1:3) * currPts + repmat(currExt(1:3,4),1,size(currPts,2));
%     allPts = [allPts currPts];
%     allColors = [allColors cloudColors{frameIDX}];
% end
