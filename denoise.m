input_img = im2double(imread('noisy_train.jpg'));
    img = input_img;
    output=img;
    n = 1;
    [row,col,~] = size(img) ;
    for k=1:5 
        grad_x = zeros(size(img));
        grad_y = zeros(size(img)); 
        [grad_x(:,:,1),grad_y(:,:,1)] = imgradientxy(img(:,:,1)) ;
        [grad_x(:,:,2),grad_y(:,:,2)] = imgradientxy(img(:,:,2)) ;
        [grad_x(:,:,3),grad_y(:,:,3)] = imgradientxy(img(:,:,3)) ;
        [x,y] = meshgrid(-1*n:n,-1*n:n);           
        for i=n+1:row-n
            for j=n+1:col-n
                Gr = [grad_x(i,j,1) ^ 2 , grad_x(i,j,1) * grad_y(i,j,1)  ; grad_x(i,j,1) * grad_y(i,j,1) , grad_y(i,j,1) ^ 2 ];
                Gg = [grad_x(i,j,2) ^ 2 , grad_x(i,j,2) * grad_y(i,j,2)  ; grad_x(i,j,2) * grad_y(i,j,2) , grad_y(i,j,2) ^ 2 ];
                Gb = [grad_x(i,j,3) ^ 2 , grad_x(i,j,3) * grad_y(i,j,3)  ; grad_x(i,j,3) * grad_y(i,j,3) , grad_y(i,j,3) ^ 2 ];
                G_sigma = Gr + Gg + Gb;    
                [V1,D1] = eig(G_sigma);
                [D,order] = sort(diag(D1), 'ascend') ;
                V = V1(:, order) ;
                largest1 = D(1) ;
                largest2 = D(2) ;
                T = 1/sqrt(1+largest1+largest2) * (V(:,1) * V(:,1)')  + 1/(1+largest1+largest2) * (V(:,2) * V(:,2)') ;
                T_inv = inv(T) ;
            
                gauss = gauss_orien(x,y,T_inv,3);
                gauss = gauss/sum(sum(gauss));
                local_img = img(i-n:i+n,j-n:j+n,:);
                for m  = 1:3
                    co = conv2(local_img(:,:,m),gauss,'same');
                    img(i,j,m) =  co(n+1,n+1);
                end
                
            end
        end
        
        ssim(img,output)
        output = img;
    end
    
    figure ; subplot(1,2,1) ; imshow(input_img) ; subplot(1,2,2) ; imshow(img,[]);
    function[out]= gauss_orien(x,y,T,t)
    out  =  exp(-1*(x.*x*T(1,1)+x.*y*T(2,1)+x.*y*T(1,2)+y.*y*T(2,2))/(4*t))/4*pi*t;    
    end