dataDir = fullfile('/media/hia/DATA/segmentation','data','test');
testDir = fullfile('/media/hia/DATA/segmentation','data','ground_truth');

%{
imageNames = {'ISIC_0000025','ISIC_0000031','ISIC_0000043','ISIC_0000049'};
%}

imagesPattern = fullfile(dataDir, '*.jpg');
images = dir(imagesPattern);
numImages = length(images);


scores = zeros(numImages,6);

for i = 300:400
    disp(i);
    imageName = images(i).name(1:end-4);
    thisImage = fullfile(dataDir, strcat(imageName, '.jpg'));
    im = imread(thisImage);
    outs = illuminati(im);
    [h,w,n] = size(outs);
    for out = 1:n
        gt = imread(fullfile(testDir, strcat(imageName, '_segmentation.png')));
        %subplot(n,2,out*2-1), imshow(gt);
        %subplot(n,2,out*2), imshow(outs(:,:,out));
        scores(i,out) = scores(i,out) + testrep(outs(:,:,out), gt);
    end
end
scores = sum(scores, 1);
scores=scores/100;
x={'red','green','blue','hue','saturation','value'};
figure
bar(scores)
set(gca,'xticklabel',x.')
title('Comparing different theresholding channels');
xlabel('Channels')
ylabel('Score')
disp(scores);