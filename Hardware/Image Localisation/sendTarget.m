function [] = sendTarget(theta,distance,mouse)
% Function to transmit wirelessly the target data. The only way to
% communicate to the Thymio is through the Aseba IDE. You have to click the
% transmitted variable, type the data to send and then click okay. This
% function takes the target data, puts them into the system's clipboard,
% uses the system robot to then open the necessary window in the Aseba IDE,
% pastes the data and clicks okay. It is custom made for the Mac laptop
% used to run the image localisation algorithm due to knowing the exact
% pixel location of the buttons. It needs to be changed if it is to be used
% on any other computer.

% Convert target data to strings
theta_string = num2str(ceil(theta));
distance_string = num2str(ceil(distance));

% Append data in format that the Thymio will interpret
transmitted_string = [theta_string,',',distance_string];

import java.awt.Robot
import java.awt.event.*

% Insert string into clipboard
clipboard('copy', transmitted_string);

% Open the transmission window in Aseba IDE
mouse.mouseMove(1100,414);
pause(0.25)
mouse.mousePress(InputEvent.BUTTON1_MASK);
mouse.mouseRelease(InputEvent.BUTTON1_MASK);
pause(0.25)
mouse.mousePress(InputEvent.BUTTON1_MASK);
mouse.mouseRelease(InputEvent.BUTTON1_MASK);
mouse.mousePress(InputEvent.BUTTON1_MASK);
mouse.mouseRelease(InputEvent.BUTTON1_MASK);

pause(0.25)

% Right click in area to put data
mouse.mouseMove(572,432);
pause(0.3)
mouse.mousePress(InputEvent.BUTTON3_MASK);
mouse.mouseRelease(InputEvent.BUTTON3_MASK);

% Paste the target data stored in the clipboard into transmission area
mouse.mouseMove(600,535);
pause(0.3)
mouse.mousePress(InputEvent.BUTTON1_MASK);
mouse.mouseRelease(InputEvent.BUTTON1_MASK);

% Click send
mouse.mouseMove(824,450);
pause(0.3)
mouse.mousePress(InputEvent.BUTTON1_MASK);
mouse.mouseRelease(InputEvent.BUTTON1_MASK);
end

