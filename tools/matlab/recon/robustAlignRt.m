function [Rt,conf] = robustAlignRt(SIFTdesFrom,SIFTdesTo,SIFT3DFrom,SIFT3DTo,dense3DFrom,dense3DTo,matchSIFT_threshold,matchSIFT_ratioTest,error3D_threshold,DISicpRatioTol,DISicpDisTol)

% use RANSAC to initialize the matching, then use ICP with smart rejection to refine it.
% test if ICP destroys the correpondence, if not, then use the ICP transformation, other wise, use RANSA

if ~exist('matchSIFT_threshold','var')
    matchSIFT_threshold = 4;
end

if ~exist('matchSIFT_ratioTest','var')
    matchSIFT_ratioTest = 0.9^2;
end

if ~exist('error3D_threshold','var')
    error3D_threshold = 0.05;
end

if ~exist('SmartRejection','var')
    SmartRejection = 2;
end

if ~exist('DISicpRatioTol','var')
    DISicpRatioTol = 2;
end

if ~exist('DISicpDisTol','var')
    DISicpDisTol = 0.1;
end

[matchPointsID_x, matchPointsID_y] = matchSIFTdesImagesBidirectional(SIFTdesFrom, SIFTdesTo,matchSIFT_ratioTest);

if length(matchPointsID_x)<matchSIFT_threshold
    numinlier = 0;
    nummatch  = 0;
    fprintf('robustAlignRt:RANSAC %s\n', 'RANSAC fails to find enough inliers');
else
    [RtRANSAC, inliers] = ransacfitRt([SIFT3DFrom(:,matchPointsID_x); SIFT3DTo(:,matchPointsID_y)], error3D_threshold, 0);
    if length(inliers)<matchSIFT_threshold
       numinlier = 0;
       nummatch  = 0;
       fprintf('robustAlignRt:RANSAC %s\n', 'RANSAC fails to find enough inliers');
    else
        % ICP refinement to get a new RtRANSAC
        maxnumofP = 10000;
        maxPoint1 = min(maxnumofP,size(dense3DFrom,2));
        rand1 = randperm(size(dense3DFrom,2));
        maxPoint2 = min(maxnumofP,size(dense3DTo,2));
        rand2 = randperm(size(dense3DTo,2));

        [TR, TT, ER, maxD] = icp(dense3DFrom(:,rand1(1:maxPoint1)),transformPointCloud(dense3DTo(:,rand2(1:maxPoint2)),RtRANSAC),'Matching','kDtree','SmartRejection',SmartRejection);

        %[TR, TT, ER, maxD] = icp(dense3DFrom,transformPointCloud(dense3DTo,RtRANSAC),'Matching','kDtree','SmartRejection',SmartRejection);
        % test if the icp destroy everything by checking SIFT distance
        SIFTfrom  = SIFT3DFrom(:,matchPointsID_x(inliers));
        SIFTto    = SIFT3DTo(:,matchPointsID_y(inliers));
        SIFTransac= transformPointCloud(SIFTto,RtRANSAC);
        SIFTicp   = transformPointCloud(SIFTto,mulRt([TR TT],RtRANSAC));

        DISransac = mean(sqrt(sum((SIFTfrom-SIFTransac).^2,1)));
        DISicp    = mean(sqrt(sum((SIFTfrom-SIFTicp   ).^2,1)));
        numinlier = length(inliers);
        nummatch  = length(matchPointsID_x);
        
        fprintf('RANSAC %f (%d/%d=%f) => ICP %f (%f=>%f, %f=>%f)\n',DISransac,length(inliers),length(matchPointsID_x),length(inliers)/length(matchPointsID_x),DISicp,ER(1),ER(end),maxD(1),maxD(end));
        if DISicp>DISransac && (DISicp > DISransac*DISicpRatioTol || DISicp > DISicpDisTol)
            Rt = RtRANSAC;
            disp('ICP is bad. Only using RANSAC');
        else
            Rt = mulRt([TR TT],RtRANSAC);
        end
    end
end



conf = struct('numinlier',numinlier,'nummatch',nummatch);


% if length(inliers)<matchSIFT_threshold||length(matchPointsID_x)<matchSIFT_threshold
%     Rt = mulRt([TR TT],RtRANSAC);
%     disp('RANSAC is bad. Only using ICP');
% end

