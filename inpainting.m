% parameters used in inpainting are
% t, filter_size
inp_image = im2double(imread("parrot.jpg"));
mask = im2double(imread('parrot_mask.jpg'));
q=10; %% Filter_size
t=2;

    img = padarray(inp_image,[q,q],0,'both');

    [row,col,~] = size(inp_image) ;
    maskbinary = imbinarize(mask,0.5);


    out =  img;
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
        
        ss=ssim(img,out);
        if ss>0.9999
           break;
        end
        out = img;

    end
        img = img(q+1:row+q,q+1:col+q,:);
        figure ; subplot(1,2,1) ; imshow(inp_image) ; subplot(1,2,2) ; imshow(img,[]);
    function[out]= gauss_orien(x,y,T,t)
    out  =  exp(-1*(x.*x*T(1,1)+x.*y*T(2,1)+x.*y*T(1,2)+y.*y*T(2,2))/(4*t))/4*pi*t;    
    end
