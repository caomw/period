

list1 = dir('../../../data/test/glue/seq01/results.1/*.png');
list3 = dir('../../../data/test/glue/seq01/results.3/*.png');

for i = 1:length(list1)
    im1 = strcat('../../../data/test/glue/seq01/results.1/',list1(i).name);
    im3 = strcat('../../../data/test/glue/seq01/results.3/',list3(i).name);
    combined = cat(2,imread(im1),imread(im3));
    imwrite(combined,strcat('seq01.',list1(i).name));
end














