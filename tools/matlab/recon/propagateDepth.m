function newD = propagateDepth(D)
% Every pixel now has a depth value if there exists depth in a local 3x3
% region around the pixel

D_layered = repmat(D,1,1,9);
D_layered(:,:,1) = imfilter(D,[1 0 0; 0 0 0; 0 0 0]);
D_layered(:,:,2) = imfilter(D,[0 1 0; 0 0 0; 0 0 0]);
D_layered(:,:,3) = imfilter(D,[0 0 1; 0 0 0; 0 0 0]);
D_layered(:,:,4) = imfilter(D,[0 0 0; 1 0 0; 0 0 0]);
D_layered(:,:,5) = imfilter(D,[0 0 0; 0 1 0; 0 0 0]);
D_layered(:,:,6) = imfilter(D,[0 0 0; 0 0 1; 0 0 0]);
D_layered(:,:,7) = imfilter(D,[0 0 0; 0 0 0; 1 0 0]);
D_layered(:,:,8) = imfilter(D,[0 0 0; 0 0 0; 0 1 0]);
D_layered(:,:,9) = imfilter(D,[0 0 0; 0 0 0; 0 0 1]);
maxD = max(D_layered,[],3);
newD = D;
newD(find(D == 0)) = maxD(find(D == 0));

end