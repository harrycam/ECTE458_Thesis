% Helper function to reset robot in different positions if desired

function in = ResetFcn(in)

global x_pos y_pos ang_start counter use_fuzzy

counter = counter + 1;

if counter >= 0
    use_fuzzy = 0;
end

x_pos = 3.5;

y_pos = 1;

ang_start = pi/4;
end