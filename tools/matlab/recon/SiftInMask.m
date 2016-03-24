function indexInMask = SiftInMask(SIFTloc, mask)

ind = sub2ind(size(mask),round(SIFTloc(2,:)),round(SIFTloc(1,:)));
indexInMask = mask(ind)>0;
