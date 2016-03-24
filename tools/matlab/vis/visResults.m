

list3 = dir('../../../data/test/glue/seq04/results.3/*.png');
list4 = dir('../../../data/test/glue/seq04/results.4/*.png');
list6 = dir('../../../data/test/glue/seq04/results.6/*.png');

for i = 1:length(list3)
    im3 = strcat('../../../data/test/glue/seq04/results.3/',list3(i).name);
    im4 = strcat('../../../data/test/glue/seq04/results.4/',list4(i).name);
    im6 = strcat('../../../data/test/glue/seq04/results.6/',list6(i).name);
    combined = cat(2,imread(im3),imread(im4),imread(im6));
    imwrite(combined,strcat('seq04.',list3(i).name));
end














