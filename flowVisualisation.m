

img = im2double(imread('download.jpg'));
result_img = flowV(img, 10);

function [ img ] = flowV(img,no_of_iterations)

    [r,c,chan] = size(img) ;
    flow = zeros([r,c,2]) ;    
    fl = zeros([r,c,4]) ;
    origin = [c/2,r/2];
    
    for i=1:r       
        for j=1:c
           % if(j <= c/2)
            %   flow(i,j,:) = [3;0];
           %end
           %if(j > c/2)
            %  flow(i,j,:) = [0;3];
           %end
            
             vec_2_o =  (origin - [j,i] + 0.00000001);
            normalized = vec_2_o/norm(vec_2_o);
             flow(i,j,:) = 3*normalized.';


            A = reshape(flow(i,j,:),2,1) ; 
            B = reshape((A*A')/norm(A),4,1) ;
            fl(i,j,:) = B ; 
        end
    end
    
    inp = img; 

    for i = 1:no_of_iterations
            
        img = im2double(img) ; 
        
        [dx1,dy1] = imgradientxy(img(:,:,1)) ; 
        [dx2,dy2] = imgradientxy(img(:,:,2)) ; 
        [dx3,dy3] = imgradientxy(img(:,:,3)) ; 

        [dxx1,dxy1] = imgradientxy(dx1) ; 
        [dyx1,dyy1] = imgradientxy(dy1) ; 

        [dxx2,dxy2] = imgradientxy(dx2) ;
        [dyx2,dyy2] = imgradientxy(dy2) ; 

        [dxx3,dxy3] = imgradientxy(dx3) ;
        [dyx3,dyy3] = imgradientxy(dy3) ; 

        % Hessian matrices
        H1 = cat(3,dxx1,dyx1,dxy1,dyy1) ;
        H2 = cat(3,dxx2,dyx2,dxy2,dyy2) ;
        H3 = cat(3,dxx3,dyx3,dxy3,dyy3) ;


        img(:,:,1) = img(:,:,1) + 0.01.*(fl(:,:,1).*H1(:,:,1) + fl(:,:,2).*H1(:,:,3) + fl(:,:,3).*H1(:,:,2) + fl(:,:,4).*H1(:,:,4)) ; 
        img(:,:,2) = img(:,:,2) + 0.01.*(fl(:,:,1).*H2(:,:,1) + fl(:,:,2).*H2(:,:,3) + fl(:,:,3).*H2(:,:,2) + fl(:,:,4).*H2(:,:,4)) ; 
        img(:,:,3) = img(:,:,3) + 0.01.*(fl(:,:,1).*H3(:,:,1) + fl(:,:,2).*H3(:,:,3) + fl(:,:,3).*H3(:,:,2) + fl(:,:,4).*H3(:,:,4)) ; 

    end
    %[r, c] = size(flow);

    %[X, Y] = meshgrid(1:c, 1:r);
    %figure;
    out = img ; 
    figure ; subplot(1,2,1) ; imshow(inp) ; subplot(1,2,2) ; imshow(out,[]);
    
end

    