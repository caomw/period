function [valid, P3D] = get3DforSIFT(XYZcam,loc)

        Xcam = XYZcam(:,:,1);
        Ycam = XYZcam(:,:,2);
        Zcam = XYZcam(:,:,3);
        validM = logical(XYZcam(:,:,4));
        ind = sub2ind([size(XYZcam,1),size(XYZcam,2)],round(loc(2,:)),round(loc(1,:)));
        valid = validM(ind);
        ind = ind(valid);
        P3D = [Xcam(ind); Ycam(ind); Zcam(ind)];

end