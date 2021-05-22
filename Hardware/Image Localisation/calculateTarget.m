function [theta, distance] = calculateTarget(cam,threshhold,counter)

   %Function fits lines and circles using Hough transform to then calculate
   %the angle and distance between the target and robot
   
   % Acquire a single image.
   rgbImage = snapshot(cam);

   % Crop image to interested area
   rgbImage = rgbImage(50:1000,480:1400,:);
   
   % Write image to file for creating video at the end
   filename = ['images/',num2str(counter),'.png'];
   imwrite(rgbImage,filename);
   
   % Convert image to gray scale and apply b/w threshhold
   I = rgb2gray(rgbImage);
   I(I<=threshhold) = 0;
   I(I>threshhold) = 255;
   
   % Apply edge filter and calculate hough transform
   BW = edge(I,'canny');
   [H,T,R] = hough(BW);
   P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
   
   % Fit lines to image using Hough transform
   lines = houghlines(BW,T,R,P,'FillGap',40,'MinLength',20);
   max_len = 0;
   min_dist = Inf;
   
   % Use Hough transform to fit circles of radisu 10-20 pixels in image
   [centers, radii, metric] = imfindcircles(I,[10,20],'ObjectPolarity','dark','Sensitivity',0.99);
   
   % Try to calculate the angle and distance to target, if failed, return
   % error
   try
   
       centersStrong = centers(1:1,:);
       radiiStrong = radii(1:1);
       metricStrong = metric(1:1);
   
       for k = 1:length(lines)
           xy = [lines(k).point1; lines(k).point2]; % Create line segments
           
           % Determine the endpoints of the longest line segment
           len = norm(lines(k).point1 - lines(k).point2);
           if ( len > max_len)
               max_len = len;
               xy_long = xy;
           end
           
           % Determine which line has least distance to circle, this is the
           % robot line
           cent_dist_vector = [xy(1,:);centersStrong];
           dist = pdist(cent_dist_vector, 'euclidean');
           if (dist < min_dist)
               robot_line = k;
               circle_pt = xy(1,:);
               min_dist = pdist(cent_dist_vector);
           end
           
           % Need to calculate the distance to circle for both points of
           % the line
           cent_dist_vector = [xy(2,:);centersStrong];
           dist = pdist(cent_dist_vector, 'euclidean');
           if (dist < min_dist)
               robot_line = k;
               circle_pt = xy(2,:);
               min_dist = pdist(cent_dist_vector);
           end
           
       end
       
       % Compute target line and midpoint
       xy_target = [lines(mod(robot_line+1,2)).point1; lines(mod(robot_line+1,2)).point2];
       target_mp = (xy_target(1,:)+xy_target(2,:))./2;
       
       % Compute robot line and midpoint
       xy_robot = [lines(robot_line).point1; lines(robot_line).point2];
       robot_mp = (xy_robot(1,:)+xy_robot(2,:))./2;
       
       % Calculate joining line between robot and target midpoint
       joining_line = [target_mp;robot_mp];
       
       v_target = [target_mp,0] - [robot_mp,0];% Calculate joining vector
       v_robot = [circle_pt,0] - [robot_mp,0];% Calculate robot line vector
       
       % Calculate distance and angle to target using these two vectors
       % through vector geometry
       theta = atan2d(v_target(1)*v_robot(2)-v_target(2)*v_robot(1),v_target(1)*v_robot(1)+v_target(2)*v_robot(2));
       distance = pdist([target_mp;robot_mp]);
       
       % Rescale the angle to between (-180, 180];
       if theta > 180
           theta = theta - 360;
       end
       
       theta % Print theta for debugging purposes
   catch error % If error, displat and return -1 to indicate error to above layer
       % Error reading will not be sent to Thymio
       disp('Error calculating theta');
       distance = -1;
       theta = -1;
   end
end

