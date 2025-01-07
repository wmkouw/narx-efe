function xk1 = pendulumStateFcn(xk, u)
%% Discrete-time nonlinear dynamic model of a pendulum at time k
%
% 2 states (xk): 
%   angle (theta): when 0, pendulum is at upright position
%   angular velocity (theta_dot): when positive, pendulum moves anti-clockwisely
% 
% 1 inputs: (uk)
%   force (F): when positive, force pushes cart to right 
%
% 2 outputs: (yk)
%   same as states (i.e. all the states are measureable)
%
% xk1 is the states at time k+1.
%
% Copyright 2016 The MathWorks, Inc.

%#codegen

uk = u(1);
Ts = u(2);
xk1 = pendulumDT0(xk, uk, Ts);
