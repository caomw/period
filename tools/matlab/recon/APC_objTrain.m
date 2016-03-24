function objectModel = APC_objTrain(objnames,datapath,objectModel)


numofTraningFrame =110;


for objectID =length(objectModel)+1:length(objnames);
    mkdir(fullfile(datapath,objnames{objectID},'pts'));
    sift = [];
    pcDense =[];
    extrinsicsC2W =[];
    frameIds =[];
    cnt =0;
    for frameid =1:numofTraningFrame
        %{
        usethisFrame = (objectID<=length(objectModel)&&~ismember(frameid,objectModel{objectID}.frameIds))...
                       ||objectID>length(objectModel);
        if objectID<=length(objectModel)
           sift = objectModel{objectID}.sift;
           pcDense = {objectModel{objectID}.XYZ};
        end
        %}
        
        if 1%exist(fullfile(datapath,objnames{objectID},[num2str(frameid) '.jpg']),'file')
            
            % read image from training : need to change to multiview   
            filepath = fullfile(datapath,objnames{objectID},[num2str(frameid) '.mat']);
            %[imTrain,XYZcam] = readData_realsense(filepath);
            [~,XYZcameraTestUncalib,cameraRGBImage_data] = readData_realsense(filepath);
         
            % Calibrate data from sensor
            [imTrain, XYZcam] = calibrateData_realsense(cameraRGBImage_data, XYZcameraTestUncalib);


            %objLabel = imread(fullfile(datapath,objnames{objectID},[num2str(frameid) '.jpg']));
            selectedMask = XYZcam(:,:,4)>0&XYZcam(:,:,3)<0.4;%&double(objLabel(:,:,1))>100;
            [loc,des] = up_sift(single(rgb2gray(imTrain)));     

            inMask = SiftInMask(loc,selectedMask);

            loc = loc(:,inMask);
            des = des(:,inMask);
            [valid, P3D] = get3DforSIFT(XYZcam,loc);  
            loc = loc(:,valid);
            des = des(:,valid);
            pc = get3Dpoints(XYZcam,selectedMask);
            % align different frames
            if frameid ==1
               Rt = [eye(3),[0;0;0]];
               validthisframe = 1;
            else

               error3D_threshold = 0.005;
               matchSIFT_threshold = 8;
               matchSIFT_ratioTest = 0.7^2;
               [Rt,NumMatch] = robustAlignRt([sift.des],des,...
                                                      [sift.P3D],P3D,...
                                                      cell2mat(pcDense),pc,...
                                                      matchSIFT_threshold,matchSIFT_ratioTest,error3D_threshold);
                
                validthisframe = NumMatch.numinlier/NumMatch.nummatch>0.7;
            end
            if 1%validthisframe
                cnt = cnt+1;
                extrinsicsC2W{cnt} = Rt;
                sift(cnt).loc = loc;
                sift(cnt).des = des;
                sift(cnt).P3D = transformPointCloud(P3D,extrinsicsC2W{cnt});
                pcDense{cnt} = transformPointCloud(pc,extrinsicsC2W{cnt});
                frameIds(cnt) = frameid;
                pts = pcDense{cnt};
                rgb = get3Dpoints(imTrain, selectedMask);
                % output ModelPLy
                %points2ply( fullfile(datapath,objnames{objectID},'pts',['s_' num2str(frameid) '.ply']), pts, repmat(rand(1,3)*255,[length(pts),1]));
                points2ply( fullfile(datapath,objnames{objectID},'pts',['c_' num2str(frameid) '.ply']), pts, rgb)
            end
            
            
        end
    end


objectModel{objectID}.sift.loc = [sift.loc];
objectModel{objectID}.sift.des = [sift.des];
objectModel{objectID}.sift.P3D = [sift.P3D];
objectModel{objectID}.XYZ =  cell2mat(pcDense);
objectModel{objectID}.extrinsicsC2W = extrinsicsC2W;
objectModel{objectID}.frameIds =frameIds;

% find 
filepath = fullfile(datapath,objnames{objectID},[num2str(1) '.mat']);
[imTrain,XYZcam] = readData_realsense(filepath);
[Rtilt,R] = rectify(objectModel{objectID}.XYZ);
%{
imshow(imTrain);
 hold on; 
 plot(sift(cnt).loc(1,:),sift(cnt).loc(2,:),'xr')

pts = [reshape(XYZcam(:,:,1),[],1),reshape(XYZcam(:,:,2),[],1),reshape(XYZcam(:,:,3),[],1)];
rgb = [reshape(imTrain(:,:,1),[],1),reshape(imTrain(:,:,2),[],1),reshape(imTrain(:,:,3),[],1)];
vis_point_cloud([R*pts']',double(rgb)/255,5,10000);
vis_point_cloud([pts']',double(rgb)/255,5,100000);
points2ply('test.ply',pts,rgb)
%}
objectModel{objectID}.XYZ      = [R*objectModel{objectID}.XYZ];
objectModel{objectID}.sift.P3D = [R*objectModel{objectID}.sift.P3D];
objectModel{objectID}.R        = R;
T = 2;
inrange = objectModel{objectID}.XYZ(1,:)>prctile(objectModel{objectID}.XYZ(1,:),T)&...
          objectModel{objectID}.XYZ(1,:)<prctile(objectModel{objectID}.XYZ(1,:),100-T)&...
          objectModel{objectID}.XYZ(2,:)>prctile(objectModel{objectID}.XYZ(2,:),T)&...
          objectModel{objectID}.XYZ(2,:)<prctile(objectModel{objectID}.XYZ(2,:),100-T)&...
          objectModel{objectID}.XYZ(3,:)>prctile(objectModel{objectID}.XYZ(3,:),T)&...
          objectModel{objectID}.XYZ(3,:)<prctile(objectModel{objectID}.XYZ(3,:),100-T);
objectModel{objectID}.XYZ =  objectModel{objectID}.XYZ(:,find(inrange));

% zero mean the model
center = mean(objectModel{objectID}.XYZ,2);
objectModel{objectID}.XYZ = bsxfun(@minus,objectModel{objectID}.XYZ,center);
objectModel{objectID}.sift.P3D = bsxfun(@minus,objectModel{objectID}.sift.P3D,center);
objectModel{objectID}.name = objnames{objectID};
obj = objectModel{objectID};
save(fullfile(datapath,objnames{objectID},'obj.mat'),'obj')
%
%vis_point_cloud(objectModel{objectID}.XYZ',[],5,10000);
end
