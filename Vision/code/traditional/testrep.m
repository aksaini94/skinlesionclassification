function score = testrep( segmented, im )

[h,w] = size(im);

score = sum(sum(xor(im, segmented)));

score = score/(h*w);

end

