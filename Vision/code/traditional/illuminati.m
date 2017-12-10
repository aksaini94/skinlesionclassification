function segmented = illuminati(im)

[h, w, n] = size(im);

representations = {@(x)x, @rgb2hsv};

[~, reps] = size(representations);

segmented = zeros(h, w , n*reps);

for rep = 1:reps
    image = representations{rep}(im);
    for channel = 1:n
        bw = im2bw(imadjust(image(:,:,channel)));
        if rep ~= 2 || channel ~= 2
            bw = imcomplement(bw);
        end
        segmented(:,:,(rep-1)*n+channel) = bwareaopen(bw, 50);
    end
end
        
    
        
 
