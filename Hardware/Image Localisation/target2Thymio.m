clear, clc;

cam = webcam('C922 Pro Stream Webcam'); % Initiate webcam access

threshhold = 30 ; % Set b/w threshhold

initial_click = 1; % Flag to indicate first run of loop

key = zeros(50); % Initialise keyboard array
counter = 1; % Set counter to save videos to 1

while key(41) ~= 1 % While ESC key not pressed, run
   
    [theta, distance] = calculateTarget(cam,threshhold,counter); % Calculate the target data
    
    % Import mouse and keyboard controlling library
    import java.awt.Robot
    import java.awt.event.*
    mouse = Robot;
    
    if initial_click == 1 % If first loop, minimise MATLAB
        mouse.mouseMove(33,33);
        mouse.mousePress(InputEvent.BUTTON1_MASK);
        mouse.mouseRelease(InputEvent.BUTTON1_MASK);
        initial_click = 0;
    end
    
    
    if distance ~= -1 % If no error calculating target data, send to Thymio
        sendTarget(theta,distance,mouse);
    end
    
    [~,~,key] = KbCheck; % Check what keys are pressed
    counter = counter + 1; % Increment image numbering counter
    
end

videoname = ['tunedVideos/',datestr(datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss')),'.avi'];
video = VideoWriter(videoname); %create the video object
open(video); %open the file for writing

for ii=1:counter-1 %where N is the number of images
  filename = ['images/',num2str(ii),'.png'];
  I = imread(filename); %read the next image
  writeVideo(video,I); %write the image to file
  writeVideo(video,I); %write the image to file
  writeVideo(video,I); %write the image to file
end

close(video); %close the file