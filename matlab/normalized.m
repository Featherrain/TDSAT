function Y = normalized(X, bandwise)
if ~exist('bandwise', 'var')
    bandwise = 0;
end

if bandwise
    Y = zeros(size(X));
    for i = 1:size(X,3)
        Xi = X(:,:,i);
        maxX = max(Xi(:));
        minX = min(Xi(:));
        Y(:,:,i) = (Xi-minX)/(maxX-minX);        
    end
else
    maxX = max(X(:));
    minX = min(X(:));
    Y    = (X-minX)/(maxX-minX);
end
end