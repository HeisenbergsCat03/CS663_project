%%% We are performing restoration on image whose 50% pixels are gone using 
%%% inpainting and get the original image back


% parameters used in inpainting are
% t, filter_size


inp_image = im2double(imread("redparrot.png"));
q=10; %% Filter_size
t=2;

%%% Removing the pixels and creating the appropriate mask
inp_image = inp_image(1:2:end,1:2:end,:);
[row,col,channels] = size(inp_image);
out = zeros(size(inp_image));
mask = ones(row,col);

for i = 1:row
    for j = 1:col
        if(mod(i,2) == 1 && mod(j,2) == 0)
            continue;
        end
        
        if(mod(i,2) == 0 && mod(j,2) == 1)
            continue;
        end
        mask(i,j) = 0;
        out(i,j,:) = inp_image(i,j,:);
    end
end

    img = padarray(out,[q,q],0,'both');

    [row,col,~] = size(out) ;
    maskbinary = imbinarize(mask,0.5);


    output =  img;
    for k=1:10    %% Maximum iterations
        
        grad_x = zeros(size(img));
        grad_y = zeros(size(img)); 
        [grad_x(:,:,1),grad_y(:,:,1)] = imgradientxy(img(:,:,1)) ;
        [grad_x(:,:,2),grad_y(:,:,2)] = imgradientxy(img(:,:,2)) ;
        [grad_x(:,:,3),grad_y(:,:,3)] = imgradientxy(img(:,:,3)) ;
        [x,y] = meshgrid(-1*q:q,-1*q:q);

            
        for i=q+1:row+q
            for j=q+1:col+q
        
                if (maskbinary(i-q,j-q) < 1) 
                    continue;
                end
                Gr = [grad_x(i,j,1) ^ 2 , grad_x(i,j,1) * grad_y(i,j,1)  ; grad_x(i,j,1) * grad_y(i,j,1) , grad_y(i,j,1) ^ 2 ];
                Gg = [grad_x(i,j,2) ^ 2 , grad_x(i,j,2) * grad_y(i,j,2)  ; grad_x(i,j,2) * grad_y(i,j,2) , grad_y(i,j,2) ^ 2 ];
                Gb = [grad_x(i,j,3) ^ 2 , grad_x(i,j,3) * grad_y(i,j,3)  ; grad_x(i,j,3) * grad_y(i,j,3) , grad_y(i,j,3) ^ 2 ];
                
                G_sigma = Gr + Gg + Gb;
                filter = fspecial('gaussian',3,1);
                G = imfilter(G_sigma,filter);
                [V1,D1] = eig(G);
                [D,order] = sort(diag(D1), 'ascend') ;
                V = V1(:, order) ;
                largest1 = D(1) ;
                largest2 = D(2) ;
                T = 1/sqrt(1+largest1+largest2) * (V(:,1) * V(:,1)')  + 1/(1+largest1+largest2) * (V(:,2) * V(:,2)') ;
                T_inv = inv(T) ;
            
                gauss = gauss_orien(x,y,T_inv,t);
                gauss = gauss/sum(sum(gauss));
                local_img = img(i-q:i+q,j-q:j+q,:);
                for m = 1:3
                    co = conv2(local_img(:,:,m),gauss,'same');
                    img(i,j,m) =  co(q+1,q+1);
                end
                
            end
        end
        
        ss=ssim(img,output);
        if ss>0.9999
           break;
        end
        output = img;

    end
        img = img(q+1:row+q,q+1:col+q,:);
figure ; subplot(1,3,1) ; imshow(inp_image) ; subplot(1,3,2) ; imshow(out); subplot(1,3,3) ; imshow(img);
    function[out]= gauss_orien(x,y,T,t)
    out  =  exp(-1*(x.*x*T(1,1)+x.*y*T(2,1)+x.*y*T(1,2)+y.*y*T(2,2))/(4*t))/4*pi*t;    
    end
