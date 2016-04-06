function filledDepth = fillHolesDepth(depth)
% Every pixel now has a depth value if there exists depth in a local 3x3
% region around the pixel

depthLayers = repmat(depth,1,1,9);
depthLayers(:,:,1) = imfilter(depth,[1 0 0; 0 0 0; 0 0 0]);
depthLayers(:,:,2) = imfilter(depth,[0 1 0; 0 0 0; 0 0 0]);
depthLayers(:,:,3) = imfilter(depth,[0 0 1; 0 0 0; 0 0 0]);
depthLayers(:,:,4) = imfilter(depth,[0 0 0; 1 0 0; 0 0 0]);
depthLayers(:,:,5) = imfilter(depth,[0 0 0; 0 1 0; 0 0 0]);
depthLayers(:,:,6) = imfilter(depth,[0 0 0; 0 0 1; 0 0 0]);
depthLayers(:,:,7) = imfilter(depth,[0 0 0; 0 0 0; 1 0 0]);
depthLayers(:,:,8) = imfilter(depth,[0 0 0; 0 0 0; 0 1 0]);
depthLayers(:,:,9) = imfilter(depth,[0 0 0; 0 0 0; 0 0 1]);
localMaxDepth = max(depthLayers,[],3);
filledDepth = depth;
filledDepth(find(depth == 0)) = localMaxDepth(find(depth == 0));

end