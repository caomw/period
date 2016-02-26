function XYZ = get3Dpoints(XYZcam, mask)
if size(XYZcam,3)==3
    ind = mask;
    XYZ = reshape(XYZcam([ind ind ind]),[],3)';

else
    ind = XYZcam(:,:,4)>0 & mask;
    XYZ = reshape(XYZcam([ind ind ind ind]),[],4)';
end
XYZ = XYZ(1:3,:);